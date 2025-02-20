echo $'\n==== Stopping MongoDB ===='
    if pgrep mongod > /dev/null; then
        echo "🛑 Stopping MongoDB..."
        mongosh admin --eval "db.shutdownServer()"
        sleep 2  # Give MongoDB time to shut down
        if pgrep mongod > /dev/null; then
            echo "⚠️ Warning: MongoDB did not stop correctly, forcing shutdown..."
            pkill -9 mongod
        fi
        echo "✅ MongoDB stopped successfully."
    else
        echo "✅ MongoDB was not running."
    fi