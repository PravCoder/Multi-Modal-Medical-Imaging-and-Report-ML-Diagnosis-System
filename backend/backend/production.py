import os
from pathlib import Path
from .settings import *  # Import everything from base settings

# Override settings for production environment

# SECURITY WARNING: keep the secret key used in production secret!
# read from environment variable (set this on your server)
SECRET_KEY = os.environ.get("SECRET_KEY")

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Set allowed hosts to your domain and/or server IP
# You'll set this as an environment variable on your server
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")

# Since you don't have a database, use dummy backend
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",  # Make sure BASE_DIR is defined!
    }
}

# Static files configuration (for your CSS, JS, images - NOT your model file)
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# Add React build files to static files directories
# REACT_BUILD_DIR = os.path.join(BASE_DIR, '../frontend/dist')

# STATICFILES_DIRS = [
#     REACT_BUILD_DIR,
#     # Add other static directories if needed
# ]

# Template configuration to serve React's index.html
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [REACT_BUILD_DIR],  # Add React build directory
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# WhiteNoise configuration for serving static files
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',  # Add this
    # ... rest of your middleware ...
]

# Whitenoise configuration
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'


# python3 manage.py runserver --settings=your_project_name.production
