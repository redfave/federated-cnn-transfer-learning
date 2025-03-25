from os import path

from dotenv import dotenv_values, load_dotenv

APP_CONFIG = dotenv_values(
    dotenv_path=path.join(".", "config.env"),  # laods from root dir of the app
    verbose=True,
)

print(APP_CONFIG)
