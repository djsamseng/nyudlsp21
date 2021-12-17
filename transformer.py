
import time

import numpy as np
import torch
import torchtext
import tqdm

device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

Softargmax = torch.nn.Softmax

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim_model, num_heads, p, dim_input=None):
        super(MultiHeadAttention, self).__init__()
        self.dim_model = dim_model
        self.num_heads = num_heads
        if dim_input is None:
            dim_xq = dim_xk = dim_xv = dim_model
        else:
            dim_xq = dim_xk = dim_xv = dim_input

        assert dim_model % self.num_heads == 0
        self.dim_k = dim_model // self.num_heads

        self.W_query = torch.nn.Linear(dim_xq, dim_model, bias=False)
        self.W_key = torch.nn.Linear(dim_xk, dim_model, bias=False)
        self.W_value = torch.nn.Linear(dim_xv, dim_model, bias=False)

        self.W_hidden = torch.nn.Linear(dim_model, dim_model)

    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.shape[0]
        key_length = K.shape[-2]

        Q = Q / np.sqrt(self.dim_k) # (bs, num_heads, query_length, dim_per_head)
        # t = num_keys = num_values = dim_model
        # Transpose K from (bs, num_heads, key_length, query_length)
        # to (bs, num_heads, query_length, key_length)
        # matmul does matrix multiplication for the last 2 dimensions in a for loop for the first dimensions
        scores = torch.matmul(Q, K.transpose(2,3)) # (bs, num_heads, query_length, key_length)

        A = torch.nn.Sigmoid()(scores)
        #A = Softargmax(dim=-1)(scores) # (bs, num_heads, query_length, key_length)

        H = torch.matmul(A, V) # (bs, num_heads, num_quries, query_dim)

        return H, A

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into [heads, depth]
        return transpose to put into shape [batch_size, num_heads, seq_length, dim_key]
        """
        return x.view(batch_size, -1, self.num_heads, self.dim_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine heads again to get [batch_size, seq_length, num_heads * dim_key]
        """
        x = x.transpose(1, 2)
        x = x.contiguous()
        x = x.view(batch_size, -1, self.num_heads * self.dim_k)
        return x

    def forward(self, X_q, X_k, X_v):
        # X_q (bs, seq_length, dim_model=embedding size)
        batch_size, seq_length, dim = X_q.size()
        Q = self.W_query(X_q) # (bs, num_heads, seq_length, dim_model)
        K = self.W_key(X_k)
        V = self.W_value(X_v)
        # Above we fed out entire word embedding into a fc layer
        # The output of which should have encoded enough information into
        # each section such that each head has enough to know about every word that is needed
        # Since scaled_dot_product_attention calculates a softmax we want multiple heads
        # to be able to focus on multiple parts of the sequence
        # A better way to do this might be to have 1 head and replace the softargmax
        # with just a sigmoid
        Q = self.split_heads(Q, batch_size) # (bs, num_heads, seq_length, dim_per_head)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Calculate attention for each head
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)
        # H_cat (bs, num_heads, seq_length, dim_model/num_heads)
        # A (bs, num_heads, seq_length, seq_length)
        # A how much attention to give to each other element

        H_cat = self.group_heads(H_cat, batch_size) # (bs, q_length, dim_model)

        H = self.W_hidden(H_cat) # (bs, q_length, dim)

        return H, A

class CNN(torch.nn.Module):
    def __init__(self, dim_model, hidden_dim, p):
        super(CNN, self).__init__()
        self.k1convL1 = torch.nn.Linear(dim_model, hidden_dim)
        self.k1convL2 = torch.nn.Linear(hidden_dim, dim_model)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x

class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_model, num_heads, conv_hidden_dim, p=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(dim_model, num_heads, p)
        self.cnn = CNN(dim_model, hidden_dim=conv_hidden_dim, p=p)
        # Normalize over the last dimension expected to be of size dim_model
        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=dim_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=dim_model, eps=1e-6)

    def forward(self, x):
        attn_output, unused_attn = self.mha(x, x, x)

        out1 = self.layernorm1(x + attn_output)

        cnn_output = self.cnn(out1)
        out2 = self.layernorm2(out1 + cnn_output)

        return out2

def create_sinusoidal_embeddings(nb_p, dim, E):
    """
    E(p, 2i) = sin(p / ( 10000^(2i/d)  ))
    E(p, 2i+1) = cos(p / ( 10000^(2i/d) ))
    """
    E.detach_()
    E.requires_grad = False
    theta = np.array([
        [ p / np.power(10000, 2 * (j // 2) / dim) for j in range(dim) ] for p in range(nb_p)
    ])
    E[:, 0::2] = torch.FloatTensor(np.sin(theta[:, 0::2]))
    E[:, 1::2] = torch.FloatTensor(np.cos(theta[:, 1::2]))
    E.detach_()
    E.requires_grad = False
    E = E.to(device)
    return E

class Embeddings(torch.nn.Module):
    def __init__(self, dim_model, vocab_size, max_position_embeddings, p):
        super(Embeddings, self).__init__()
        self.word_embeddings = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim_model, padding_idx=1)
        self.position_embeddings = torch.nn.Embedding(num_embeddings=max_position_embeddings, embedding_dim=dim_model)
        create_sinusoidal_embeddings(
            nb_p=max_position_embeddings,
            dim=dim_model,
            E=self.position_embeddings.weight
        )
        self.layer_norm = torch.nn.LayerNorm(dim_model, eps=1e-12)

    def forward(self, input_ids):
        # input_ids are (bs, num_words) where each word is an int
        seq_length = input_ids.shape[1]
        position_ids = torch.arange(end=seq_length, dtype=torch.long, device=device)

        # 0 1 2 3 4 ... num_words
        # 0 1 2 3 4 ... num_words
        # ... bs times
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids) #(bs, num_words)

        # Get embedding for each input id
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings) # (bs, sequence_length=number of words)

        return embeddings

class Encoder(torch.nn.Module):
    def __init__(self, num_layers, dim_model, num_heads, ff_hidden_dim, input_vocab_size, maximum_position_encoding, p=0.1):
        super(Encoder, self).__init__()

        self.dim_model = dim_model
        self.num_layers = num_layers

        self.embedding = Embeddings(dim_model, vocab_size=input_vocab_size, max_position_embeddings=maximum_position_encoding, p=p)
        self.enc_layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.enc_layers.append(EncoderLayer(dim_model, num_heads, conv_hidden_dim=ff_hidden_dim, p=p))

    def forward(self, x):
        # (bs, seq_length=num_words) each element is an int corresponding to the word
        x = self.embedding(x)
        # (bs, seq_length=num_words, feature_size=dim_model)
        # each word has a vector of floats (size dim_model) that describes the word and
        # has its position encoded with the sinusoidal

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x

class TransformerClassifier(torch.nn.Module):
    def __init__(self, num_layers, dim_model, num_heads, conv_hidden_dim, input_vocab_size, num_answers) -> None:
        super(TransformerClassifier, self).__init__()
        self.encoder = Encoder(num_layers, dim_model, num_heads,
            ff_hidden_dim=conv_hidden_dim,
            input_vocab_size=input_vocab_size,
            maximum_position_encoding=10_000, p=0.1)
        self.dense = torch.nn.Linear(dim_model, num_answers)

    def forward(self, x):
        x = self.encoder(x)

        x, _ = torch.max(x, dim=1)
        x = self.dense(x)
        return x

def train(model, optimizer, loss_fn, train_loader, num_epochs):
    model.train()
    for train_idx in range(num_epochs):
        train_itr = iter(train_loader)
        train_acc = 0
        losses = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(train_itr):
            BX = batch.text.to(device)
            BY = batch.label.to(device)

            out = model(BX)
            loss = loss_fn(out, BY)
            model.zero_grad()
            loss.backward()
            losses += loss.item()
            optimizer.step()

            train_acc += (out.argmax(1) == BY).cpu().numpy().mean()
            num_batches += 1

            if batch_idx % 100 == 0:
                print("Train itr:", train_idx, "Batch:", batch_idx, "loss:", losses/num_batches, "accuracy:", train_acc/num_batches)

        # TODO:?
        print("Train itr:", train_idx, "loss:", losses/num_batches, "accuracy:", train_acc/num_batches)


BATCH_SIZE = 164

def startup():
    max_len = 200
    text = torchtext.legacy.data.Field(sequential=True, fix_length=max_len, batch_first=True, lower=True, dtype=torch.long)
    label = torchtext.legacy.data.LabelField(sequential=False, dtype=torch.long)
    t0 = time.time()
    ds_train, ds_test = torchtext.legacy.datasets.IMDB.splits(text, label)
    ds_train, ds_valid = ds_train.split(0.9)
    print("Download:", time.time()-t0)

    print("Num train:", len(ds_train), "Num valid:", len(ds_valid), "Num test:", len(ds_test))
    print("train.fields", ds_train.fields)
    print("Text:", ds_train[0].text, "Label:", ds_train[0].label)
    num_words = 50_000
    text.build_vocab(ds_train, max_size=num_words)
    print("Build text vocab:", time.time()-t0)
    label.build_vocab(ds_train)
    print("Build label vocab:", time.time()-t0)
    vocab = text.vocab
    # Converts strings into integers represening the words, one number for each word
    train_loader, valid_loader, test_loader = torchtext.legacy.data.BucketIterator.splits(
        (ds_train, ds_valid, ds_test), batch_size=BATCH_SIZE, sort_key=lambda x: len(x.text), repeat=False
    )
    return (train_loader,)

def transformer_run(train_loader):
    train_example = next(iter(train_loader))
    print("train_loader iter:",
        "Text:", train_example.text.shape, train_example.text[0][:5], "...",
        "Label:", train_example.label.shape, train_example.label[0])
    # 1 head, sigmoid, 10 epochs loss:0.1091 accuracy:0.9658
    # 2 heads, softargmax, 10 epochs loss:0.1608 accuracy:0.9455
    model = TransformerClassifier(num_layers=1, dim_model=32, num_heads=1,
        conv_hidden_dim=128, input_vocab_size=50_002, num_answers=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    train(model, optimizer, loss_fn=torch.nn.functional.cross_entropy, train_loader=train_loader,
        num_epochs=10)

def transformer_main():
    train_loader, = startup()
    transformer_run(train_loader)


def multiheadattension_check():
    mha = MultiHeadAttention(dim_model=512, num_heads=8, p=0)

    def print_out(Q, K, V):
        print("Query:", Q)
        print("Key:", K)
        print("Value:", V)
        temp_out, temp_attn = mha.scaled_dot_product_attention(Q, K, V)
        print("Attention:", temp_attn)
        print("Output:", temp_out)
        return temp_out, temp_attn

    # 1 batch, 1 head, 4 keys (t), query dim 3
    test_K = torch.tensor([[
        [[10, 0, 0],
        [ 0,10, 0],
        [ 0, 0,10],
        [ 0, 0,10]]
    ]]).float()

    # 1 batch, 1 head, 4 keys (t), query dim 3
    test_V = torch.tensor([[
        [[   1,0,0],
        [  10,0,0],
        [ 100,5,0],
        [1000,6,0]]
    ]]).float()

    # Query is 3 dimensional
    # we have 4 keys which are 3 dimensional
    # 1 batch, 1 head, 2 queries, query of dim 3
    test_Q = torch.tensor([[
        [[0, 10, 0],
         [10, 1, 0]
        ]
    ]]).float()
    # the first query matches the second value
    # thus we see the output is very close to [10,0,0]
    # the second query matches the first value and a little of the second value
    # thus we see the output is very close to [1, 0, 0] with a little [10,0,0]
    H, A = print_out(test_Q, test_K, test_V)
    torch.testing.assert_allclose(A.shape, torch.Size([1, 1, 2, 4]))
    expected = torch.tensor([[
        [
            [
                torch.dot(test_K[0,0,0], test_Q[0,0,0]),
                torch.dot(test_K[0,0,1], test_Q[0,0,0]),
                torch.dot(test_K[0,0,2], test_Q[0,0,0]),
                torch.dot(test_K[0,0,3], test_Q[0,0,0]),
            ],
            [
                torch.dot(test_K[0,0,0], test_Q[0,0,1]),
                torch.dot(test_K[0,0,1], test_Q[0,0,1]),
                torch.dot(test_K[0,0,2], test_Q[0,0,1]),
                torch.dot(test_K[0,0,3], test_Q[0,0,1]),
            ],
        ]
    ]])
    expected /= np.sqrt(mha.dim_model // mha.num_heads)
    print("Expected scores:", expected)
    expected = torch.nn.Softmax(dim=-1)(expected)
    torch.testing.assert_allclose(A, expected)
    # 1, 1, 2, 4 X 1, 1, 4, 3 = 1, 1, 2, 3
    expected_values = torch.matmul(A, test_V)
    torch.testing.assert_allclose(H.shape, torch.Size([1, 1, 2, 3]))
    torch.testing.assert_allclose(H, expected_values)

if __name__ == "__main__":
    multiheadattension_check()
    transformer_main()