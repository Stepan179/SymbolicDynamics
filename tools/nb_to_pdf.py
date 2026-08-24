"""PDF-копия ноутбука: текст, код, вывод и графики. Сам .ipynb не изменяется.

Генерирует собственный .tex (PT Serif / PT Mono) и компилирует xelatex; pandoc и
браузер не требуются.

Буква ё в тексте, коде и выводе заменяется на е (в графиках остаётся как есть).

Input:  один или несколько .ipynb
Output: <каталог ноутбука>/pdf/<имя>.pdf
Usage:  python3 tools/nb_to_pdf.py analysis/notebooks/*.ipynb
"""
import sys, os, re, shutil, base64, subprocess, tempfile, nbformat

XELATEX = (shutil.which("xelatex")
           or os.path.expanduser("~/Library/TinyTeX/bin/universal-darwin/xelatex"))

ESC = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
       '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
def noyo(s): return s.replace('ё', 'е').replace('Ё', 'Е')
def esc(s): return ''.join(ESC.get(c, c) for c in noyo(s))

GREEK = {'Δ': r'$\Delta$', 'δ': r'$\delta$', 'σ': r'$\sigma$', 'μ': r'$\mu$', 'γ': r'$\gamma$',
         'π': r'$\pi$', 'ξ': r'$\xi$', 'φ': r'$\varphi$', 'χ': r'$\chi$', 'τ': r'$\tau$',
         'ρ': r'$\rho$', 'α': r'$\alpha$', 'β': r'$\beta$', 'ε': r'$\varepsilon$', 'λ': r'$\lambda$',
         'θ': r'$\theta$', '→': r'$\to$', '×': r'$\times$', '≈': r'$\approx$', '≥': r'$\ge$',
         '≤': r'$\le$', '≪': r'$\ll$', '∈': r'$\in$', '√': r'$\sqrt{\ }$', '∑': r'$\sum$',
         '²': r'\textsuperscript{2}', '³': r'\textsuperscript{3}', '·': r'$\cdot$', '±': r'$\pm$',
         '∞': r'$\infty$', '⊗': r'$\otimes$', '∫': r'$\int$', '≠': r'$\ne$', '⟹': r'$\Rightarrow$'}
def greek(s): return ''.join(GREEK.get(c, c) for c in s)

def inln(s):
    out = ''
    for p in re.split(r'(`[^`]*`|\*\*[^*]+\*\*)', s):
        if len(p) >= 2 and p[0] == '`' and p[-1] == '`':
            out += r'\texttt{' + greek(esc(p[1:-1])) + '}'
        elif len(p) >= 4 and p.startswith('**') and p.endswith('**'):
            out += r'\textbf{' + greek(esc(p[2:-2])) + '}'
        else:
            out += greek(esc(p))
    return out

HEAD = {1: r'\section*', 2: r'\subsection*', 3: r'\subsubsection*', 4: r'\paragraph*',
        5: r'\paragraph*', 6: r'\paragraph*'}

def md_to_tex(src):
    out, lst = [], None
    def close():
        nonlocal lst
        if lst:
            out.append(r'\end{' + lst + '}')
            lst = None
    for line in src.split('\n'):
        h = re.match(r'^(#{1,6})\s+(.*)', line)
        li = re.match(r'^\s*[-*]\s+(.*)', line)
        ni = re.match(r'^\s*\d+\.\s+(.*)', line)
        if h:
            close(); out.append(HEAD[len(h.group(1))] + '{' + inln(h.group(2)) + '}')
        elif li or ni:
            want = 'itemize' if li else 'enumerate'
            if lst != want:
                close(); out.append(r'\begin{' + want + '}'); lst = want
            out.append(r'\item ' + inln((li or ni).group(1)))
        elif line.strip() == '':
            close(); out.append('')
        else:
            close(); out.append(inln(line) + r' \\')
    close()
    return '\n'.join(out)

def verb(text, rule):
    text = noyo(text).rstrip('\n')
    if not text.strip():
        return ''
    return (r'\begin{Verbatim}[frame=leftline,framerule=2pt,rulecolor=' + rule +
            r',fontsize=\small]' + '\n' + text + '\n' + r'\end{Verbatim}')

PRE = r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\usepackage{polyglossia}
\setmainlanguage{russian}\setotherlanguage{english}
\setmainfont{PT Serif}[Ligatures=TeX]
\newfontfamily\cyrillicfont{PT Serif}[Ligatures=TeX]
\setmonofont{PT Mono}\newfontfamily\cyrillicfonttt{PT Mono}
\usepackage{amsmath,amssymb}
\usepackage[a4paper,margin=1.9cm]{geometry}
\usepackage{graphicx,xcolor,fvextra}
\usepackage{enumitem}\setlist{topsep=1pt,itemsep=0pt,leftmargin=1.4em}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.45em}
\tolerance=3000 \emergencystretch=4em \hbadness=10000 \hfuzz=3pt
\fvset{breaklines=true,breakanywhere=true}
\begin{document}
'''

def convert(path):
    nb = nbformat.read(path, 4)
    name = os.path.splitext(os.path.basename(path))[0]
    tmp = tempfile.mkdtemp()
    tex = [PRE, r'\textbf{\large ' + esc(name) + r'}\par\vspace{1em}']
    img = 0
    for c in nb.cells:
        s = c.source if isinstance(c.source, str) else ''.join(c.source)
        if c.cell_type == 'markdown':
            tex.append(md_to_tex(s))
        elif c.cell_type == 'code':
            tex.append(verb(s, r'\color{blue!45}'))
            for o in c.get('outputs', []):
                t = o.get('output_type')
                dat = o.get('data', {})
                if 'image/png' in dat:
                    p = os.path.join(tmp, f'im{img}.png'); img += 1
                    open(p, 'wb').write(base64.b64decode(dat['image/png']))
                    tex.append(r'\begin{center}\includegraphics[width=0.72\linewidth]{' + p + r'}\end{center}')
                elif t == 'stream':
                    tex.append(verb(o.get('text', '')[:4000], r'\color{gray!45}'))
                elif 'text/plain' in dat:
                    txt = ''.join(dat['text/plain']) if isinstance(dat['text/plain'], list) else dat['text/plain']
                    if not re.match(r'^\s*<.*(Axes|Figure|matplotlib|seaborn).*>\s*$', txt):
                        tex.append(verb(txt[:4000], r'\color{gray!45}'))
    tex.append(r'\end{document}')
    texpath = os.path.join(tmp, name + '.tex')
    open(texpath, 'w').write('\n'.join(tex))
    env = dict(os.environ, PATH=os.path.dirname(XELATEX) + ':' + os.environ['PATH'])
    for _ in range(2):
        r = subprocess.run([XELATEX, '-interaction=nonstopmode', '-halt-on-error', name + '.tex'],
                           cwd=tmp, env=env, capture_output=True, text=True)
    outdir = os.path.join(os.path.dirname(path), 'pdf')
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(tmp, name + '.pdf')
    if os.path.exists(pdf):
        dst = os.path.join(outdir, name + '.pdf')
        subprocess.run(['cp', pdf, dst])
        print('OK ->', dst)
    else:
        print('FAIL', name)
        print(r.stdout[-2500:])

if __name__ == '__main__':
    for p in sys.argv[1:]:
        convert(p)
