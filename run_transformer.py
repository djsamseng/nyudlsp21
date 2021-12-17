import importlib

import transformer

train_loader, = transformer.startup()

while True:
    _ = input("Press enter to run loop")
    try:
        importlib.reload(transformer)
        transformer.transformer_run(train_loader)
    except Exception as e:
        print("Failed to reload:", e)