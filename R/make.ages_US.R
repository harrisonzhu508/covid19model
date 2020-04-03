library(tidyverse)
library(data.table)

indir = "~/Downloads" # path directory

## LOAD DATA FROM USA CENSUS 
# download csv from URL bellow and save into path directory
#https://data.census.gov/cedsci/table?q=age%20state&hidePreview=true&tid=ACSST1Y2018.S0101&t=Age%20and%20Sex&vintage=2018&cid=S0101_C01_001E&g=0100000US.04000.001&y=2018&moe=false&tp=false

data = read.csv(file = file.path(indir, "export.csv"), header = TRUE, sep = ",", stringsAsFactors = FALSE)

## GROUP BY 10 Y AGE BAND
df = data.table(AGE = data[which(data[,3] == "AGE"),1][-1], data[which(data[,3] == "AGE"),which(as.character(data[1,]) == "Total")][-1,]) %>%
  melt(id.vars = "AGE") %>%
  rename(STATE=variable, POP=value) %>%
  mutate(POP = as.numeric(str_replace_all(POP, "[[:punct:]]", "")), 
         AGE = ifelse(AGE == "Under 5 years", "0-5", 
                      ifelse(AGE == "85 years and over", "85+", 
                             paste(str_match(AGE, "(.*?) to")[,2], str_match(AGE, "to (.*?) years")[,2], sep="-")))) %>%
  mutate(POP = as.numeric(as.character(POP)),
         grp = rep(1:(nrow(.)/2), each = 2)) %>%
  group_by(grp) %>%
  mutate(
    AGE = paste(AGE, collapse = ":"),
    AGE = gsub("-\\d+:\\d+", "", AGE)) %>%
  mutate(POP = sum(POP)) %>%
  slice(1) %>%
  ungroup() %>%
  select(-grp)

## CONVERT FROM LONG TO WIDE FORMAT
df.wide = df %>%
  reshape2::dcast(STATE ~ AGE, value.var="POP")
write.csv(df.wide,file.path(indir,"ages.csv"), row.names = FALSE)
