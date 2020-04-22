
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org" 
       options(repos=r)
})

## # Install devtools from CRAN
## install.packages("devtools")
## 
## # Use devtools to install hdf5r and loomR from GitHub
## devtools::install_github(repo = "hhoeflin/hdf5r")
## devtools::install_github(repo = "mojaveazure/loomR", ref = "develop")
## 
## if (!requireNamespace("BiocManager", quietly = TRUE))
## 	    install.packages("BiocManager")
## 
## BiocManager::install("scater")

#install.packages('Seurat', repos='http://cran.us.r-project.org')
install.packages('Seurat')
