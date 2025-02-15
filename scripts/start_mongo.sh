#!/bin/bash

if ! pgrep mongod > /dev/null
then
    echo "🚀 Starting MongoDB..."
    mongod --dbpath ~/MongoDB_Thesis --logpath ~/MongoDB_Thesis/mongo.log --fork
    echo "✅ MongoDB has been started."
else
    echo "✅ MongoDB is already running."
fi
