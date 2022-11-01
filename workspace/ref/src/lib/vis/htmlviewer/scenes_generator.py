''' html generator for scenes

author
    Zhangsihao Yang

date
    2022-0427

name convention
    lc = list caption   
    ld = list description
    lt = list type
    lpd = link path in docker image space
    lu = list url
    lid = list of list of dict
    npp = number per page
    otf = output folder
    pf = path of file
'''
import math
import os
import sys

pf = os.path.abspath(__file__)
pf = os.path.dirname(pf)
sys.path.append(pf)

from util import forward_backward

html_head = (
    '<html lang="en" >\n'
    '<head>\n'
    '\t<meta charset="UTF-8">\n'
    '\t<title>ZYViewer</title>\n'
    '\t<script src="https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/three.min.js"></script>\n'
    '\t<script src="https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/controls/OrbitControls.js"></script>\n'
    '\t<script src="https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/loaders/LoaderSupport.js"></script>\n'
    '\t<script src="https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/loaders/OBJLoader2.js"></script>\n'
    '\t<script src="https://r105.threejsfundamentals.org/threejs/resources/threejs/r105/js/loaders/OBJLoader.js"></script>\n'
    '\t<script type="text/javascript"  src="./script.js"></script>\n'
    '<style>\n'
    '\t* {\n'
    '\t\tbox-sizing: border-box;\n'
    '\t\t-moz-box-sizing: border-box;\n'
    '\t}\n'
    '\tbody {\n'
    '\t\tbackground-color: #fff;\n'
    '\t\tcolor: #444;\n'
    '\t}\n'
    '\ta {\n'
    '\t\tcolor: #08f;\n'
    '\t}\n'
    '\t#content {\n'
    '\t\tposition: absolute;\n'
    '\t\ttop: 0; width: 100%;\n'
    '\t\tz-index: 1;\n'
    '\t\tpadding: 3em 0 0 0;\n'
    '\t}\n'
    '\t#c {\n'
    '\t\tposition: absolute;\n'
    '\t\tleft: 0;\n'
    '\t\twidth: 100%;\n'
    '\t\theight: 100%;\n'
    '\t}\n'
    '\t.list-item {\n'
    '\t\tdisplay: inline-block;\n'
    '\t\tmargin: 1em;\n'
    '\t\tpadding: 1em;\n'
    '\t\tbox-shadow: 1px 2px 4px 0px rgba(0,0,0,0.25);\n'
    '\t}\n'
    '\t.list-item > div:nth-child(1) {\n'
    '\t\twidth: 200px;\n'
    '\t\theight: 200px;\n'
    '\t\toverflow: hidden;\n'
    '\t\ttext-overflow:ellipsis;\n'
    '\t\toverflow-wrap: break-word;\n'
    '\t}\n'
    '\t.list-item > div:nth-child(2) {\n'
    '\t\tcolor: #888;\n'
    '\t\tfont-family: sans-serif;\n'
    '\t\tfont-size: large;\n'
    '\t\twidth: 200px;\n'
    '\t\tmargin-top: 0.5em;\n'
    '\t}\n'
    '</style>\n'
    '</head>\n'
    '<body>\n'
)
html_tail = (
    # '\t\t<script type=\"text/javascript\">\n'
    # '$(\'.entry button\').click(function() {{\n'
    # '$(this).parents(\'.entry\').find(\'.folder\').\n'
    # "slideToggle(1000);}})\n"
    # "\t\t</script>\n\t</body>\n</html>"
    '</body>\n'
    '</html>\n'
)

class ScenesGenerator():
    def __init__(
        self, root, lld, otf, npp
    ):
        '''
        args
            root : the location in the 
                real system of the data 
                folder
            lid : see the main function
                for an example
            otf : the folder in the 
                vurtial system to save
                the html files
            npp : number of item per 
                page
        '''
        self.root = root
        self.lld = lld
        self.otf = otf
        self.npp = npp

        self._prepare()

        self.func()

        self._generate_index()

    def _generate_index(self):
        fmain = open(
            os.path.join(
                self.otf,
                'index.html'
            ), 'w'
        )
        fmain.write(
            f"<html><head><meta http-equiv=\"refresh\" "
            f"content=\"0; url=0.html\" /></head></html>"
        )

    def _prepare(self):
        # create folder to save
        os.makedirs(
            self.otf, exist_ok=True
        )
        # copy the script
        sfp = os.path.join(
            pf, 'script.js'
        )
        tfp = os.path.join(
            self.otf, 'script.js'
        )
        if not os.path.exists(tfp):
            os.system(
                f'cp {sfp} {tfp}'
            )
        # make the folder link
        lpd = os.path.join(
            self.otf, 'data'
        )
        if not os.path.islink(lpd):
            os.system(
                f'ln -s {self.root} '
                f'{lpd}'
            )

    @staticmethod
    def _edit_lu(lu):
        # add 'data/' before every url
        for i, u in enumerate(lu):
            lu[i] = 'data/' + lu[i]
        return lu

    def func(self):
        # Total number of pages.
        tot_page_num = math.ceil(
            len(
                self.lld
            ) * 1.0 / self.npp
        )

        # go over the list of file
        ld = []
        lc = []
        lu = []
        lt = []
        for i in range(
            len(self.lld)
        ):
            p = i // self.npp

            # process the first item 
            # in this page
            if i % self.npp == 0:
                out_filename = \
                  os.path.join(
                      self.otf, 
                      f'{p}.html'
                  )
                fout = open(
                    out_filename, 'w'
                )
                fout.write(html_head)
                fout.write(
                    '\t<canvas id="c"></canvas>\n'
                    '\t<div id="content">\n'
                    '\t\t<h3>Page '
                    f'Id: {p}</h3>\n'
                )
                forward_backward(
                    fout, p,
                    tot_page_num
                )
                fout.write(
                    '\t</div>\n'
                    '\t<script type="text/javascript">\n'
                )

            print(f"Processing {i}: {i}")

            for j in range(
                len(self.lld[i])
            ):
                ld.append(
                    self.lld[
                        i
                    ][j]['desc']
                )
                lc.append(
                    self.lld[
                        i
                    ][j]['caption']
                )
                lu.append(
                    self.lld[
                        i
                    ][j]['url']
                )
                lt.append(
                    self.lld[
                        i
                    ][j]['type']
                )

            # after finishing the 
            # insertion of last item 
            # in this page
            if (
                (i + 1) % self.npp == 0
            ) or (
                (i + 1) == len(
                    self.lld
                )
            ):
                lu = self._edit_lu(lu)
                # enter the render
                fout.write(
                    '\t\tscenes_render(\n'
                    f'\t\t\t{str(ld)},\n'
                    f'\t\t\t{str(lc)},\n'
                    f'\t\t\t{str(lu)},\n'
                    f'\t\t\t{str(lt)},\n'
                    '\t\t)\n'
                )
                fout.write(
                    '\t</script>\n'
                )
                # write the tail
                fout.write(html_tail)
                # close the file
                fout.close()
                # reset the variables
                ld = []
                lc = []
                lu = []
                lt = []


def main():
    root = '/home/george/George/dataset/modelnet/off_to_obj/obj/'
    lid = [
        [
            {
                'desc': 'air0001',
                'caption': 'mesh',
                'url': 'airplane/train/airplane_0001/model.obj',
                'type': 'mesh',
            }
        ],
        [
            {
                'desc': 'air0002',
                'caption': 'mesh',
                'url': 'airplane/train/airplane_0002/model.obj',
                'type': 'mesh',
            }
        ],
        [
            {
                'desc': 'air0003',
                'caption': 'mesh',
                'url': 'airplane/train/airplane_0003/model.obj',
                'type': 'mesh',
            }
        ],
        [
            {
                'desc': 'air0004',
                'caption': 'mesh',
                'url': 'airplane/train/airplane_0004/model.obj',
                'type': 'mesh',
            }
        ],
        [
            {
                'desc': 'air0005',
                'caption': 'mesh',
                'url': 'airplane/train/airplane_0005/model.obj',
                'type': 'mesh',
            }
        ],
        [
            {
                'desc': 'air0006',
                'caption': 'mesh',
                'url': 'airplane/train/airplane_0006/model.obj',
                'type': 'mesh',
            }
        ],
    ]
    otf = '/dataset/modelnet/off_to_obj/html'
    nnp = 5
    sg = ScenesGenerator(
        root, lid, otf, nnp
    )


if __name__ == '__main__':
    main()        
