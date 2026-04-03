#!/bin/bash

# Configuration
DRIVE_DIR="HCMUS/NLP-LegalQA"
LOCAL_DIR="."
USER_EMAIL="quangminhcantho43@gmail.com"

# Note: Authentication is required on first run
# gog login $USER_EMAIL

echo "Uploading files from $LOCAL_DIR to Google Drive: $DRIVE_DIR"

# get folder id (creates if missing)
FOLDER_ID=$(gog drive search "name = '$DRIVE_DIR' and mimeType = 'application/vnd.google-apps.folder'" --account "$USER_EMAIL" --json | jq -r '.files[0].id // empty')

if [ -z "$FOLDER_ID" ]; then
    echo "Creating folder $DRIVE_DIR..."
    FOLDER_ID=$(gog drive mkdir "$DRIVE_DIR" --account "$USER_EMAIL" --json | jq -r '.id')
fi

# Upload files one by one (mimics basic sync)
find "$LOCAL_DIR" -type f | while read -r file; do
    echo "Uploading $file..."
    gog drive upload "$file" --parent "$FOLDER_ID" --account "$USER_EMAIL"
done
