#!/usr/bin/python
# -*- coding: iso-8859-1 -*-
"""
    mProfile syntax highlighting and profile results display
"""
# Based on the MoinMoin python source coloriser
#
# which is in turn based on the code from J�rgen Herman, to which the following changes where made:
#
# Mike Brown <http://skew.org/~mike/>:
# - make script callable as a CGI and a Apache handler for .py files.
#
# Christopher Arndt <http://chrisarndt.de>:
# - make script usable as a module
# - use class tags and style sheet instead of <style> tags
# - when called as a script, add HTML header and footer
#
# #
# 
# David Baddeley david_baddeley <at> yahoo.com.au
# converted to display mProfile output:
# - added code to display line times, line numbers, and to highlight the expensive lines
# - may have broken some of the original features

__version__ = '0.1'
__date__ = '2009-01-22'
__license__ = 'GPL'
__author__ = 'J�rgen Hermann, Mike Brown, Christopher Arndt, David Baddeley'


# Imports
import cgi, string, sys
from io import StringIO
import keyword, token, tokenize

try:
    from cgi import escape
except ImportError:
    from html import escape

#############################################################################
### Python Source Parser (does Highlighting)
#############################################################################

_KEYWORD = token.NT_OFFSET + 1
_TEXT    = token.NT_OFFSET + 2

_css_classes = {
    token.NUMBER:       'number',
    token.OP:           'operator',
    token.STRING:       'string',
    tokenize.COMMENT:   'comment',
    token.NAME:         'name',
    token.ERRORTOKEN:   'error',
    _KEYWORD:           'keyword',
    _TEXT:              'text',
}

_HTML_HEADER = """\
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
  "http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
  <title>%%(title)s</title>
  <meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
  <meta name="Generator" content="colorize.py (version %s)">
</head>
<body>
""" % __version__

_HTML_FOOTER = """\
</body>
</html>
"""

_STYLESHEET = """\
<style type="text/css">
pre.code {
    font-style: Lucida,"Courier New";
}

.number {
    color: #0080C0;
}
.operator {
    color: #000000;
}
.string {
    color: #008000;
}
.comment {
    color: #808080;
}
.name {
    color: #000000;
}
.error {
    color: #FF8080;
    border: solid 1.5pt #FF0000;
}
.keyword {
    color: #0000FF;
    font-weight: bold;
}
.text {
    color: #000000;
}

}


</style>

"""

class Parser:
    """ Send colored python source.
    """

    stylesheet = _STYLESHEET

    def __init__(self, raw, times, counts, thresholdT = 0.01, out=sys.stdout):
        """ Store the source text.
        """
        self.raw = raw.expandtabs().strip()
        self.out = out
        self.times = times
        self.counts = counts

        self.totTime = sum(times.values())
        self.maxTime = max(max(times.values()), 1e-3)
        self.cover_flag = False
        self.line_flag = False
        self.thresholdT = thresholdT
        

    def format(self):
        """ Parse and send the colored source.
        """
        # store line offsets in self.lines
        self.lines = [0, 0]
        pos = 0
        
        while 1:
            pos = self.raw.find('\n', pos) + 1
            if not pos: break
            self.lines.append(pos)
        self.lines.append(len(self.raw))

        # parse the source and write it
        self.pos = 0
        text = StringIO(u'' + self.raw)
        self.out.write(self.stylesheet)
        self.out.write('<pre class="code">\n')
        try:
            #tokenize.tokenize(text.readline, self)
            for tokens in tokenize.generate_tokens(text.readline):
                self(*tokens)
        except tokenize.TokenError as ex:
            msg = ex[0]
            line = ex[1][0]
            self.out.write("<h3>ERROR: %s</h3>%s\n" % (
                msg, self.raw[self.lines[line]:]))
            if self.cover_flag:
                self.out.write('</span>')
                self.cover_flag = False
        self.out.write('\n</pre>')

    def __call__(self, toktype, toktext, start, end, line):
        """ Token handler.
        """
        srow, scol = start
        erow, ecol = end
        if 0:
            print("type", toktype, token.tok_name[toktype], "text", toktext,)
            print("start", srow,scol, "end", erow,ecol, "<br>")

        # calculate new positions
        oldpos = self.pos
        newpos = self.lines[srow] + scol
        self.pos = newpos + len(toktext)

        
        if (not self.line_flag):
            #tspec = 'XXX.YYs'
            tspec =  '       ' + '       '
            if (srow in self.times.keys()):
                #print srow
                t = self.times[srow]
                gb = int(250*(1 - t/self.maxTime))
                self.out.write('<span style="background-color :#FF%02X%02X; ">' %(gb, gb))
                self.cover_flag = True

                if (t/self.maxTime > self.thresholdT):
                    tspec = '%6.2fs n=%04d' % (t, self.counts[srow])


            self.out.write('%s    %03d    ' % (tspec,srow ))
            self.line_flag = True

        
        #if scol == 0 and srow == 1:
        #    self.out.write('%03d    ' % srow)


        # handle newlines
        if toktype in [token.NEWLINE, tokenize.NL]:
            if self.cover_flag:
                self.out.write('</span>')
                self.cover_flag = False
            self.out.write('\n')
            self.line_flag = False
            #self.out.write('%03d    ' % (srow + 1))
            return

        

        # send the original whitespace, if needed
        if newpos > oldpos:
            self.out.write(self.raw[oldpos:newpos])
            #ws = self.raw[oldpos:newpos].split('\n')
            #ws1 = ws[0]
            #for i in range(len(ws[1:])):
            #    ws1 += '\n%03d    %s' % ((srow - len(ws) + i), ws[1+i])
            #self.out.write(ws1)

        # skip indenting tokens
        if toktype in [token.INDENT, token.DEDENT]:
            self.pos = newpos
            return

        # map token type to a color group
        if token.LPAR <= toktype and toktype <= token.OP:
            toktype = token.OP
        elif toktype == token.NAME and keyword.iskeyword(toktext):
            toktype = _KEYWORD
        css_class = _css_classes.get(toktype, 'text')

        # send text
        self.out.write('<span class="%s">' % (css_class,))
        self.out.write(escape(toktext))
        self.out.write('</span>')

        if toktype == tokenize.COMMENT and toktext[-1] == '\n':
            #self.out.write('%03d    ' % (srow + 1)) 
            self.line_flag = False


def colorize_file(times, counts, file=None, outstream=sys.stdout, standalone=True):
    """Convert a python source file into colorized HTML.

    Reads file and writes to outstream (default sys.stdout). file can be a
    filename or a file-like object (only the read method is used). If file is
    None, act as a filter and read from sys.stdin. If standalone is True
    (default), send a complete HTML document with header and footer. Otherwise
    only a stylesheet and a <pre> section are written.
    """

    from os.path import basename
    if hasattr(file, 'read'):
        sourcefile = file
        file = None
        try:
            filename = basename(file.name)
        except:
            filename = 'STREAM'
    elif file is not None:
        try:
            sourcefile = open(file)
            filename = basename(file)
        except IOError:
            raise "File %s unknown." % file
    else:
        sourcefile = sys.stdin
        filename = 'STDIN'
    source = sourcefile.read()

    if standalone:
        outstream.write(_HTML_HEADER % {'title': filename})
    Parser(source, times, counts, out=outstream).format()
    if standalone:
        outstream.write(_HTML_FOOTER)

    if file:
        sourcefile.close()

#if __name__ == "__main__":
#    import os
#    if os.environ.get('PATH_TRANSLATED'):
#        filepath = os.environ.get('PATH_TRANSLATED')
#        print 'Content-Type: text/html; charset="iso-8859-1"\n'
#        colorize_file(filepath)
#    elif len(sys.argv) > 1:
#        filepath = sys.argv[1]
#        colorize_file(filepath)
#    else:
#        colorize_file()
