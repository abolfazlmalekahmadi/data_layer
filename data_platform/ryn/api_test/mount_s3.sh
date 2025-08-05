#!/bin/bash

# Usage: ./mount_ceph_s3.sh <ACCESS_KEY> <SECRET_KEY> <BUCKET_NAME> <MOUNT_POINT>

ACCESS_KEY="$1"
SECRET_KEY="$2"
BUCKET_NAME="$3"
MOUNT_POINT="$4"
ENDPOINT="$5"

if [[ $# -ne 5 ]]; then
  echo "❌ Usage: $0 <ACCESS_KEY> <SECRET_KEY> <BUCKET_NAME> <MOUNT_POINT> <ENDPOINT>"
  exit 1
fi

# Create mount point if it doesn't exist
mkdir -p "$MOUNT_POINT"

# Create temporary credentials file
TEMP_PASSWD=$(mktemp)
echo "$ACCESS_KEY:$SECRET_KEY" > "$TEMP_PASSWD"
chmod 600 "$TEMP_PASSWD"

# Mount using s3fs
s3fs "$BUCKET_NAME" "$MOUNT_POINT" \
  -o passwd_file="$TEMP_PASSWD" \
  -o url="$ENDPOINT" \
  -o use_path_request_style \
  -o allow_other

MOUNT_STATUS=$?

# Optional: remove temp credentials file
rm -f "$TEMP_PASSWD"

if [[ $MOUNT_STATUS -eq 0 ]]; then
  echo "✅ Successfully mounted $BUCKET_NAME at $MOUNT_POINT"
else
  echo "❌ Mount failed"
fi

exit $MOUNT_STATUS
