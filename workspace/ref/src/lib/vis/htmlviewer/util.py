''' util for html generator

author
    Zhangsihao Yang

date 
    2022-0427

'''
def forward_backward(
    file, page_id, tot_page_num
):
    ''' insert the forward and 
    backward bottom inside the file

    args:
        file: The file that could 
            be written.
        page_id (int): The index of 
            the current page.
        tot_page_num (int): The 
            total number of pages.
    '''
    prev_id = page_id - 1
    next_id = page_id + 1
    prev_click = ''
    next_click = ''
    if prev_id < 0:
        prev_click = (
            ' onclick="return false;"'
        )
    if next_id >= tot_page_num:
        next_click = (
            ' onclick="return false;"'
        )
    file.write(
        f"\t\t<h3><a href=\"{str(0)+'.html'}\">First</a>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<a href=\"{str(prev_id)+'.html'}\"{prev_click}>Prev</a>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<a href=\"{str(next_id)+'.html'}\"{next_click}>Next</a>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<a href=\"{str(int(tot_page_num-1))+'.html'}\">Last</a>"
        f"</h3>\n"
    )
