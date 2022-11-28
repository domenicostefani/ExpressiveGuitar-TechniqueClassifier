inotifywait -m ./output/ -e create -e moved_to |
    while read dir action file; do
        echo "The file '$file' appeared in directory '$dir'"
        # do something with the file
    done
