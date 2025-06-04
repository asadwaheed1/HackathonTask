import os
import json

SETTINGS_FILE = "user_settings.json"

def load_all_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_all_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)

class UserSettings:
    def __init__(self, user_id):
        self.user_id = user_id
        self.settings = self.load_settings()

    def load_settings(self):
        all_settings = load_all_settings()
        return all_settings.get(self.user_id, {})

    def save_settings(self):
        all_settings = load_all_settings()
        all_settings[self.user_id] = self.settings
        save_all_settings(all_settings)

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value
        self.save_settings()

    def all(self):
        return self.settings
