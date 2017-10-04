from bs4 import BeautifulSoup
import common.file_tools as file_tools
import common.constants as const
import os


# convert html files under given dir to text files
def batch_html2text(from_dir, to_dir):
    # html_regext = "^(.*\.(htm|html|shtml)$)?$"
    # html_file_list = [f for f in os.listdir(from_dir) if re.match(html_regext, f)]
    html_file_list = file_tools.list_files_under_dir(from_dir, const.HTML_EXTENSIONS)
    for filename in html_file_list:
        with open(os.path.join(from_dir, filename), 'rb') as file:
            text = html2text(file.read())
            # print(text)
        # change the extension of html file to txt
        text_file_name = ".".join(filename.split('.')[0:-1] + ["txt"])
        to_path = os.path.join(to_dir, text_file_name)
        print(to_path)
        with open(to_path, 'wt', encoding='utf-8') as to_file:
            to_file.write(text)

def html2text(html):
    # code from https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python
    soup = BeautifulSoup(html, "lxml")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

# batch_html2text('D:/work/research/gan-accounting/web-crawler/data/all', 'D:/work/research/gan-accounting/data/sec/aaer')
