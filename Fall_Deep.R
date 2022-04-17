#101-109 y 203-209, 801 y 901, test 1 - 6, 06 27 35 37 39 40

path_1 <- 'Tests/'
path_2 <- 101 
path_3 <- '/Testler Export/'
path_4 <- '8' 
path_5 <- '01/Test_'
path_6 <- 1
path_7 <- '/3405'
path_8 <- '06'
path_9 <- '.txt'
path_4_1 <- '9'

minim <- 1000
maxim <- 0

anali <- matrix(nrow = 160, ncol = 1)

ind_1 <- 1
ind_2 <- 81

file_name <- "Falls.csv"
file.create(file_name)

for(i in 101:117)
{
  for(j in 1:5)
  {
    path_D <- paste(path_1,path_2,path_3,path_4,path_5,path_6,path_7,path_8,path_9, sep='')
    path_F <- paste(path_1,path_2,path_3,path_4_1,path_5,path_6,path_7,path_8,path_9, sep='')
      
    data_1 <- read.delim(path_D)
    data_2 <- read.delim(path_F)
      
    rows_1 <- nrow(data_1)
    rows_2 <- nrow(data_2)
    
    anali[ind_1] <- rows_1
    anali[ind_2] <- rows_2
    
    print(ind_1)
    ind_1 <- ind_1 + 1
    ind_2 <- ind_2 + 1
    
    rows <- min(rows_1,rows_2)
    
    if (rows < minim)
    {
      minim <- rows
    }
    
    rows <- max(rows_1,rows_2)
    
    if (rows > maxim)
    {
      maxim <- rows
    }
      
    path_6 <- path_6 + 1
    
    #///////////////////DB//////////////////////
    
    i0_1 <- floor(rows_1 * 0.3333)
    if_1 <- i0_1 + 293
    
    i0_2 <- floor(rows_2 * 0.3333)
    if_2 <- i0_2 + 293
    
    mag_1 <- matrix(nrow = 293)
    mag_2 <- matrix(nrow = 293)
    
    for (i in i0_1:(if_1-1))
      mag_1[i - i0_1 + 1, 1] <- sqrt(data_1[i,10]*data_1[i,10] + data_1[i,11]*data_1[i,11] + data_1[i,12]*data_1[i,12])
    
    for (i in i0_2:(if_2-1))
      mag_2[i - i0_2 + 1, 1] <- sqrt(data_2[i,10]*data_2[i,10] + data_2[i,11]*data_2[i,11] + data_2[i,12]*data_2[i,12])
    
    mag_1 <- c(mag_1,0)
    mag_2 <- c(mag_2,1)
    
    mag_1[is.na(mag_1)] <- 0
    mag_2[is.na(mag_2)] <- 0
    
    mag_1 <- t(mag_1)
    write.table(mag_1, file_name, sep = ",", col.names = F, append = T, row.names = F)
    
    mag_2 <- t(mag_2)
    write.table(mag_2, file_name, sep = ",", col.names = F, append = T,row.names = F)
    
    #///////////////////////////////////////////
  }
  
  path_6 <- 1
}

rm(data_1)
rm(data_2)
rm(mag_1)
rm(mag_2)

summary(anali)
anali_t <- t(anali)
