# pyme-bootstrap

This is the source for our custom bootstap theme used in the web version of PYMEAcquire. Building the css requires
npm and sass, and we cheat a bit by tracking the generated css in version control. This means that you only ever need to
run make-css.sh if you have altered pyme.scss rather than rebuilding everytime we build PYME (as we do for, e.g. c
extensions). The caveat here is that if you do alter pyme.scss you need to manually run make-css and commit the changes
to both pyme.scss and PYME/Acquire/webui/static/css/pyme-bootstrap.css
