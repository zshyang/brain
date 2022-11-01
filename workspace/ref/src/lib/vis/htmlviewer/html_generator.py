'''
Zhangsihao Yang
04/02/2022

lt = list of titles
li = list of input
otf = output folder
npp = number per page
'''
import argparse
import json
import math
import os

html_head = f"<html>\n" + \
            f"\t<head>\n" + \
            f"\t\t<title>Simple Viewer</title>\n" + \
            f"\t\t<script " + \
            f"src=\"https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/" + \
            f"jquery.min.js\"></script>\n" + \
            f"\t\t<script src=\"http://code.jquery.com/ui/1.9.2/" + \
            f"jquery-ui.js\"></script>\n" + \
            f"\t\t<script type='text/javascript' " + \
            f"src='http://www.x3dom.org/x3dom/release/x3dom.js'>" + \
            f"</script>\n" + \
            f"\t\t<script src=\"https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/three.min.js\"></script>\n" + \
            f"\t\t<script src=\"https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/controls/OrbitControls.js\"></script>\n" + \
            f"\t\t<script src=\"https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/loaders/LoaderSupport.js\"></script>\n" + \
            f"\t\t<script src=\"https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/loaders/OBJLoader2.js\"></script>\n" + \
            f"\t\t<script  type=\"text/javascript\"  src=\"./script.js\"></script>\n" + \
            f"\t\t<style>.folder {{display: none; }}</style>\n" + \
            f"\t</head>\n\t<body>\n"

html_tail = f"\t\t<script type=\"text/javascript\">" + \
            f"$(\'.entry button\').click(function() {{" + \
            f"$(this).parents(\'.entry\').find(\'.folder\')." + \
            f"slideToggle(1000);}})\n" + \
            f"\t\t</script>\n\t</body>\n</html>"

def tag_tr(i, data, table_title):
    """Convert the information into table entry to enter the html.

    Args:
        k (int): The number of the item.
        data (tuple): The input data.
        table_title (str): The string contains the title of the table.
    
    Returns:
        The table string that contains the data and table title.
    """
    s = '\t\t\t<table border="1" style="width:100%">\n' + table_title + \
        '\t\t\t\t<tr>\n'

    ts = '\t'
    s += f'{ts*5}<td>' + str(i) + '</td>\n'
    for j, d in enumerate(data):
        d_type, d_value = data[d]
        if d_type == 'img':
            s += f"<td><a href=\"{os.path.join('..', d_value)}\">" + \
                 f"<img src=\"{os.path.join('..', d_value)}\" " + \
                 f"width=\"200px\" height=\"200px\" /></a></td>"
        elif d_type == 'text':
            s += '<td>{}</td>'.format(d_value)
        elif d_type == "model":
            s += f"<td><script src=\"" + \
                 f"{d_value}\"></script></td>\n"
        elif d_type == 'obj':
            s += f"{ts*5}<td>\n" + \
                 f'{ts*6}{d_value}\n' + \
                 f'{ts*6}<canvas id="c_{i}_{j}">\n' + \
                 f"{ts*6}<script type=\"text/javascript\">\n" + \
                 f"{ts*6}main('{d_value}', '#c_{i}_{j}')\n" + \
                 f"{ts*6}</script>\n{ts*6}</canvas>\n{ts*5}</td>\n"
        else:
            s += '<td>None</td>'

    s += f'{ts*4}</tr>\n{ts*3}</table>\n'

    return s


def parse_file_names(input_file: str):
    """Parse a file that contains the list of files names.

    Args:
        input_file (str): The input file name.
    
    Returns:
        file_name_entry (list): The list that contains the list of file names.
    """
    file_name_entry =  []  # 1d list
    if input_file.endswith("txt"):
        with open(input_file, "r") as file:
            for line in file:
                if line.endswith("\n"):
                    item = line[:-1]
                else:
                    item = line
                # Add the item into entry if exists.
                if os.path.exists(item):
                    file_name_entry.append(item)
                else:
                    print(f"{item} does not exist!")
                    exit(1)
    elif input_file.endswith("json"):
        with open(input_file, "r") as file:
            file_name_entry = json.load(file)
            # Check the existance of all the files names.
            for item in file_name_entry:
                if not os.path.exists(item):
                    print(f"{item} does not exist!")
                    exit(1)
    else:
        print(f"The extension is not supported!")
        exit(1)
    return file_name_entry


def func(num_per_page, output_folder, input_file_list, input_title_list):
    """The function that would generate the html files given the inputs.
    [[entry], [entry], [entry], [entry]]. The whole list is called stack.

    Args:
        num_per_page (int): A integer controls how many items per page.
        output_folder (str): The location of the output folder.
        input_file_list (str): The string that contains the list of input files.
            The names are seperated by comma.
        input_title_list (str): The string that contains the list of titles.
            The names are seperated by comma.
    """
    # Seperate the list of input files and input titles.
    input_file_list = input_file_list.split(",")
    print(f"input_file_list: {input_file_list}")

    input_title_list = input_title_list.split(",")
    print(f"input_title_list: {input_title_list}")

    # The length of the titles and files should be equal to each other.
    if len(input_file_list) != len(input_title_list):
        print(f"The length between input file and input title is not equal")
        exit(1)
    num_entry = len(input_file_list)

    # Check the exist of the current input files.
    for input_file in input_file_list:
        if not os.path.exists(input_file):
            print(f"Error: input file {input_file} does not exist!")
            exit(1)

    if os.path.exists(output_folder):
        print(f"Warning: output folder {output_folder} exists!")
    os.makedirs(output_folder, exist_ok=True)

    # write the file header and footer
    html_head = f"<html>\n" + \
                f"\t<head>\n" + \
                f"\t\t<title>Simple Viewer</title>\n" + \
                f"\t\t<script " + \
                f"src=\"https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/" + \
                f"jquery.min.js\"></script>\n" + \
                f"\t\t<script src=\"http://code.jquery.com/ui/1.9.2/" + \
                f"jquery-ui.js\"></script>\n" + \
                f"\t\t<script type='text/javascript' " + \
                f"src='http://www.x3dom.org/x3dom/release/x3dom.js'>" + \
                f"</script>\n" + \
                f"\t\t<style>.folder {{display: none; \}}</style>\n" + \
                f"\t</head>\n\t<body>\n"
    html_tail = f"\t\t<script type=\"text/javascript\">" + \
                f"$(\'.entry button\').click(function() {{" + \
                f"$(this).parents(\'.entry\').find(\'.folder\')." + \
                f"slideToggle(1000);}})\n" + \
                f"\t\t</script>\n\t</body>\n</html>"

    table_title = f"<tr><td><b>ID</b></td><td><b>File Name</b></td>"
    for i in range(len(input_title_list)):
        table_title += f"<td><b>" + input_title_list[i] + "</b></td>"

    # Load the names of the files.
    file_name_stack = []  # 2d list
    for input_file in input_file_list:
        file_name_entry =  parse_file_names(input_file)  # 1d list
        file_name_stack.append(file_name_entry)
    length_file_name = len(file_name_stack[0])
    # Check the length of each entry is equal to each other.
    for file_name_entry in file_name_stack:
        if len(file_name_entry) != length_file_name:
            print(f"The length between files entry should be eqaul!")
            exit(1)

    # The current item number.
    k = 0

    # Total number of pages.
    tot_page_num = math.ceil(length_file_name * 1.0 / num_per_page)

    # Go over the list of file names.
    for i in range(length_file_name):
        # Process the first item in this page.
        if k % num_per_page == 0:
            out_filename = os.path.join(output_folder, 
                str(k//num_per_page)+'.html')
            fout = open(out_filename, 'w')
            fout.write(html_head)
            fout.write(f"\t\t<h3>Page Id: {k//num_per_page}</h3>\n")

            page_id = k//num_per_page
            forward_backward(fout, page_id, tot_page_num)
       
        print(f"Processing {k}: {i}")
        k = k + 1

        # Prepare the data.
        data = {}
        counter = 1
        data[0] = ("text", file_name_stack[0][i])
        for j in range(num_entry):
            data[counter] = parse_different_data_format(file_name_stack[j][i])
            counter += 1

        # Enter the table.
        fout.write('\t\t<div class="entry">')
        fout.write(tag_tr(k, data, table_title))
        fout.write('\t\t</div>')

        # After finishing the insertion of last item in this page, we insert the
        # forward and backward bottom, write the tail then close the file.
        if k % num_per_page == 0 or k == length_file_name:
            page_id = k//num_per_page - 1
            forward_backward(fout, page_id, tot_page_num)
            fout.write(html_tail)
            fout.close()


def parse_different_data_format(item):
    """Parse the file name with different data format.
    
    Args:
        item: The name of the file.
    
    Returns:
        return a tuple with a name and a text.
    """
    if item.endswith(".png") or item.endswith(".jpg"):
        data_entry = ("img", item)
    elif item.endswith(".txt"):
        ftxt = open(item)
        txt = ""
        for line in ftxt.readlines():
            txt += '<p>' + line.rstrip() + '</p>'
        txt += f"<p><a href=\"{item}\">See More</a></p>"
        data_entry = ("text", txt)
    elif item.endswith(".js"):
        data_entry = ("model", item)
    elif item.endswith(".obj"):
        data_entry = ('obj', item)
    else:
        print(f"Warning: there is no .jpg/.png/.txt file whose name "
              f"starts with {item}!")
        data_entry = ("none", "")
    return data_entry


def forward_backward(file, page_id, tot_page_num):
    """Insert the forward and backward bottom inside the file.

    Args:
        file: The file that could be written.
        page_id (int): The index of the current page.
        tot_page_num (int): The total number of pages.
    """
    prev_id = page_id - 1
    next_id = page_id + 1
    prev_click = ''
    next_click = ''
    if prev_id < 0:
        prev_click = ' onclick="return false;"'
    if next_id >= tot_page_num:
        next_click = ' onclick="return false;"'
    file.write(f"\t\t<h3><a href=\"{str(0)+'.html'}\">First</a>"
               f"&nbsp;&nbsp;&nbsp;"
               f"<a href=\"{str(prev_id)+'.html'}\"{prev_click}>Prev</a>"
               f"&nbsp;&nbsp;&nbsp;"
               f"<a href=\"{str(next_id)+'.html'}\"{next_click}>Next</a>"
               f"&nbsp;&nbsp;&nbsp;"
               f"<a href=\"{str(int(tot_page_num-1))+'.html'}\">Last</a>"
               f"</h3>\n")


def parse_args():
    """Parse the input arguments.
    """
    parser = argparse.ArgumentParser(prog="html viewer")

    parser.add_argument("--num_per_page", type=int, default=3, 
        help="The number of items per page.")
    parser.add_argument("--output_folder", type=str, default="output_html",
        help="The location to store the output html files.")
    parser.add_argument("--input_file_list", type=str, 
        default=f"code/input_folder/image_list.txt,"
                f"code/input_folder/image_list_2.txt,"
                f"code/input_folder/js_list.txt",
        help=f"The string list of the text files that contains the names of"
             f"images, text, or 3D object to be shown.")
    parser.add_argument("--input_title_list", type=str, 
        default="images,images_2,3d_model",
        help="The title to name of the list of files")

    args = parser.parse_args()

    return args


def generate(num_per_page, output_folder, input_file_list, input_title_list):
    """ Generate the HTML files given the input arguments.

    Args:
        num_per_page (int): A integer controls how many items per page.
        output_folder (str): The location of the output folder.
        input_file_list (str): The string that contains the list of input files.
            The names are seperated by comma.
        input_title_list (str): The string that contains the list of titles.
            The names are seperated by comma.
    """
    func(num_per_page, output_folder, input_file_list, input_title_list)

    fmain = open(os.path.join(output_folder, "index.html"), "w")
    fmain.write(f"<html><head><meta http-equiv=\"refresh\" "
                f"content=\"0; url=0.html\" /></head></html>")


def main():
    # Parse the input arguments.
    args = parse_args()
    num_per_page = args.num_per_page
    output_folder = args.output_folder
    input_file_list = args.input_file_list
    input_title_list = args.input_title_list

    # Call the function to generate the html files.
    func(num_per_page, output_folder, input_file_list, input_title_list)

    fmain = open(os.path.join(output_folder, "index.html"), "w")
    fmain.write(f"<html><head><meta http-equiv=\"refresh\" "
                f"content=\"0; url=0.html\" /></head></html>")


class html_generator():
    def __init__(self, lt, li, otf, npp):
        os.makedirs(otf, exist_ok=True)

        self.lt = lt
        self.li = li
        self.otf = otf
        self.npp = npp

        self._check()
        self._copy()

        self.func()

    def generate(self):
        fmain = open(
            os.path.join(self.otf, 'index.html'), 'w'
        )
        fmain.write(
            f"<html><head><meta http-equiv=\"refresh\" "
            f"content=\"0; url=0.html\" /></head></html>"
        )

    def _check(self):
        # check length
        if len(self.lt) != len(self.li):
            raise Exception(
                "Length of input and title is not equal"
            )
        # check the length of list of input
        length_file_name = len(self.li[0])
        for column in self.li:
            if len(column) != length_file_name:
                raise Exception(
                    "Length between files should be eqaul!"
                )
        self.lid0 = len(self.li)
        self.lid1 = len(self.li[0])

    def _copy(self):
        sfp = 'htmlviewer/script.js'
        tfp = os.path.join(self.otf, 'script.js')
        if not os.path.exists(tfp):
            os.system(f'cp {sfp} {tfp}')

    def func(self):
        # create table title
        table_title = f"\t\t\t\t<tr>\n" + \
                      f"\t\t\t\t\t<td><b>ID</b></td>\n"
        for i in range(len(self.lt)):
            table_title += f"\t\t\t\t\t<td><b>{self.lt[i]}</b></td>\n"
        table_title += '\t\t\t\t</tr>\n'
        
        # # The current item number.
        # k = 0

        # Total number of pages.
        tot_page_num = math.ceil(self.lid1 * 1.0 / self.npp)

        # Go over the list of file names.
        # k = 0
        for i in range(self.lid1):
            p = i // self.npp

            # Process the first item in this page.
            if i % self.npp == 0:
                out_filename = os.path.join(
                    self.otf, f'{p}.html'
                )
                fout = open(out_filename, 'w')
                fout.write(html_head)
                fout.write(f"\t\t<h3>Page Id: {p}</h3>\n")
                forward_backward(fout, p, tot_page_num)
        
            print(f"Processing {i}: {i}")
            # k = k + 1

            # Prepare the data.
            data = {}
            # data[0] = ("text", file_name_stack[0][i])
            for j in range(self.lid0):
                data[j] = parse_different_data_format(
                    self.li[j][i]
                )

            # Enter the table.
            fout.write('\t\t<div class="entry">\n')
            fout.write(tag_tr(i, data, table_title))
            fout.write('\t\t</div>\n')

            # After finishing the insertion of last item in this page, we insert the
            # forward and backward bottom, write the tail then close the file.
            if ((i + 1) % self.npp == 0) or ((i + 1) == self.lid1):
                forward_backward(fout, p, tot_page_num)
                fout.write(html_tail)
                fout.close()


if __name__ == "__main__":
    main()
    