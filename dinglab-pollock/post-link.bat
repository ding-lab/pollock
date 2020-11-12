pip install tensorflow==2.1.0
# install rpollock functions
R -e "Sys.setenv(TAR = system('which tar', intern = TRUE)); devtools::install_github('https://github.com/estorrs/rpollock')"
