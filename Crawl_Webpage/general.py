import os

# Each website is a separate project (folder)
def create_project_dir(directory):
    if not os.path.exists(directory):
        print('Creating directory ' + directory)
        os.mkdir(directory)



# Create queue and crawled files (if not created)
def create_data_files(project_name, base_url):
    queue_file = os.path.join(project_name, 'queue.txt')
    crawled_file = os.path.join(project_name, 'crawled.txt')
    if not os.path.isfile(queue_file):
        write_file(queue_file, base_url)
    if not os.path.isfile(crawled_file):
        write_file(crawled_file, '')




# Create a new file
def write_file(path, data):
    with open(path, 'w') as file:
        file.write(data)




# Add data onto an existing file
def append_to_file(path, data):
    with open(path, 'a') as file:
        file.write(data + '\n')




# Delete the contents of a file
def delete_file_contents(path):
    with open(path, 'w') as file:
        pass




# Read a file and convert each line to set items
def file_to_set(file_name):
    results = set()
    with open(file_name,'rt') as file:
        for line in file:
            results.add(line.replace('\n',''))
        return results




# Iterate through a set, each item will be a line in a file
def set_to_file(set_links, file_name):
    with open(file_name, 'w') as f:
        for link in sorted(set_links):
            append_to_file(file_name, link)

