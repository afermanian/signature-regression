setwd("~/Documents/Thèse/Model selection/Code")

library(fda)
df<-daily$tempav
head(df)
summary(df)
write.csv(df,file="daily_temp.csv")

annualprec = log10(apply(daily$precav,2,sum))
annualprec
write.csv(annualprec,file="annual_prec.csv")
