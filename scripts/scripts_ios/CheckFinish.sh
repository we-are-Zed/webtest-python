#!/bin/bash

echo "Waiting for webtest to finish..."

while [ -e "webtest_running" ]; do
    sleep 1
done

echo "webtest has finished."
