import subprocess

filename="articles.txt"

file = open(filename, "r")
str_article = file.read()
# remove whitespaces
str_art2 = ''.join(str_article.split())
urls = str_art2.split(',')

for url in urls:
    print(subprocess.call(["scrape", url, "--html", "--no-images"]))

print("DONE!!DONE!!DONE!!")
