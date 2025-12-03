import os


def file_name_to_key_name(name):
    return name.replace(" ", "_").replace("-", "_").upper()


class Secrets(object):
    _secrets = dict[str, str]()

    def __init__(self, secrets_path):
        if os.path.exists(secrets_path):
            secrets = os.listdir(secrets_path)
        else:
            secrets = []
        for secret_file_name in secrets:
            full_secret_file_path = os.path.join(secrets_path, secret_file_name)
            try:
                with open(full_secret_file_path, encoding="utf-8") as secret_file:
                    key = file_name_to_key_name(secret_file_name)
                    self._secrets[key] = secret_file.read().strip()
            except UnicodeDecodeError:
                with open(full_secret_file_path, encoding="utf-16") as secret_file:
                    key = file_name_to_key_name(secret_file_name)
                    self._secrets[key] = secret_file.read().strip()

    def __call__(self, key, default=None):
        return self.get_value(key, default)

    def get_value(self, key, default=None) -> str:
        if key in self._secrets:
            return self._secrets[key]
        elif default is not None:
            return default
        else:
            raise ValueError(f"secret key {key} not found")
