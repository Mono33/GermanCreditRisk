#Author : Louis Mono
#Capstone2 :German Credit Risk
#repository link : https://github.com/Mono33/GermanCreditRisk

####################################################################################################################################################

#This in a R code which generates overall accuracy of Machine Learning algorithms used to solve classification problem for the german credit scoring.

#In my repository, except this GermanCredit_Rcode, you have the following files: 
#- PDFReport_GermanRisk.pdf : German credit risk report in pdf format ( download it on your pc)
#- Report in rmd format here: https://beta.rstudioconnect.com/content/5328/Report_GermanRisk.html ( Report_GermanRisk.Rmd is the rmd file that generated it )

#but you also find:
#- PDFReport_GermanRisk.Rmd : .Rmd file that generates my report in PDF format
#- all the *.PNG files that i used in the Methods and Analysis section 
#- Install-doSNOW-parallel-DeLuxe.R , performed by  Tobias Kind (2015) To resolve the summary.connection(connection) : invalid connection

##I.---------------------------------Section I: Introduction ---------------------------------------------------------------------------------------

#text on pdf and .rmd reports

##II. ------------------------------- Section 2: Dataset  ----------------------------------------------------------

#2.1. Overview --------------------------------------------------------------------------------------------------------------

#load libraries
library(tidyverse)
library(rchallenge)
library(caret)
library(RColorBrewer)
library(reshape2)
library(lattice)
library (rpart)
library(rpart.plot) 
library(rattle)
library(ROCR)
library(ggpubr)
library(ggthemes)
library(randomForest)
library(Information)
library(VIM)
library(Boruta)
library(e1071)
library(gridExtra)
library(lars)
library(glmnet)
library(kableExtra)
library(doSNOW)
library(doParallel)


#load data

data("german")

#get class/glimpse of data

class(german)
glimpse(german)  #description of all features are available in pdf and rmd reports



# 2.2. Data exploration ---------------------------------------------------------------------------------

#summary statistics
summary(german)


#outcome, y : Class ----
##################

#i create Class.prop, an object of Class 'tbl_df', 'tbl' and 'data.frame'.It contains the calculated relative frequencies or proportions for the levels Bad/Good of the credit worthiness in the german credit data.
Class.prop <- german %>% 
  count(Class) %>% 
  mutate(perc = n / nrow(german)) 

# bar plot of credit worthiness
Class.prop %>%
  ggplot(aes(x=Class,y= perc,fill=Class))+
  geom_bar(stat="identity") +
  labs(title="bar plot",
       subtitle = "Credit worthiness",
       caption=" source: german credit data") +
  geom_text(aes(label=scales::percent(perc)), position = position_stack(vjust = 1.01))+
  scale_y_continuous(labels = scales::percent)+
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))



# quantitative attributes : Amount , Age, Duration ,Installment Rate, etc ----
#######################################################################

#client's Age

ggplot(melt(german[,c(5,21)]), aes(x = variable, y = value, fill = Class)) + 
  geom_boxplot() +
  xlab("Class") +
  ylab("Age") +
  labs(title="Box plot", subtitle="client's age grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# Duration in months

avg.duration <- german %>%
  select(Duration, Class) %>%
  group_by(Class) %>%
  summarise(m=mean(Duration))

german%>% 
  ggplot(aes(Duration))+
  geom_density(aes(fill=Class),alpha=0.7) + 
  geom_vline(data=avg.duration,aes(xintercept= m , colour= Class), lty = 4 ,size=2)+
  labs(title="Density plot", 
       subtitle="Duration in months grouped by credit worthiness",
       caption="Source: german credit data",
       x="Duration",
       fill="Class") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))

ggplot(reshape2::melt(german[,c(1,21)]), aes(x = variable, y = value, fill = Class)) + 
  geom_boxplot() +
  xlab("Class") +
  ylab("Duration") +
  labs(title="Box plot", subtitle="Duration in months grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# credit Amount

avg.amount <- german %>%
  select(Amount, Class) %>%
  group_by(Class) %>%
  summarise(m=mean(Amount))


german%>% 
  ggplot(aes(Amount))+
  geom_density(aes(fill=Class),alpha=0.7) + 
  geom_vline(data=avg.amount,aes(xintercept= m , colour= Class), lty = 4 ,size=2)+
  labs(title="Density plot", 
       subtitle="credit amount grouped by credit worthiness",
       caption="Source: german credit data",
       x="Amount",
       fill="Class") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


ggplot(reshape2::melt(german[,c(2,21)]), aes(x = variable, y = value, fill = Class)) + 
  geom_boxplot() +
  xlab("Class") +
  ylab("Amount") +
  labs(title="Box plot", subtitle="credit amount grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))



# Installment rate

german %>% 
  ggplot(aes(InstallmentRatePercentage, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge")+
  labs(title="bar plot", subtitle="InstallmentRatePercentage grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


ggplot(reshape2::melt(german[,c(3,21)]), aes(x = variable, y = value, fill = Class)) + 
  geom_boxplot() +
  xlab("Class") +
  ylab("Install_rate_perc") +
  labs(title="Box plot", subtitle="InstallmentRatePercentage grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# NumberExistingCredits

# we just report here barplot/boxplot for the attribute NumberExistingCredits,
# but we show descriptive statistics including also NumberPeopleMaintenance and  ResidenceDuration. To produce their barplot and boxplot , just repeat the code below and replace with their variable name.

german %>% 
  ggplot(aes(NumberExistingCredits, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge")+
  labs(title="bar plot", subtitle="Number of existing credits grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


ggplot(reshape2::melt(german[,c(6,21)]), aes(x = variable, y = value, fill = Class)) + 
  geom_boxplot() +
  xlab("Class") +
  ylab("n_credits") +
  labs(title="Box plot", subtitle="Number of existing credits grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))

# Compare mean, median and sd of two Class  for each of these attributes

german%>% group_by(Class) %>% 
  summarize(ncredits.mean=mean(NumberExistingCredits), 
            ncredits.median=median(NumberExistingCredits), 
            ncredits.sd = sd(NumberExistingCredits))

german%>% group_by(Class) %>% 
  summarize(resid_dur.mean=mean(ResidenceDuration), 
            resid_dur.median=median(ResidenceDuration), 
            resid_dur.sd = sd(ResidenceDuration))

german%>% group_by(Class) %>% 
  summarize(people_maint.mean=mean(NumberPeopleMaintenance), 
            people_maint.median=median(NumberPeopleMaintenance), 
            people_maint.sd = sd(NumberPeopleMaintenance))

# i create a dataframe "get_num" which contains all numerical attributes plus the response variable. Then, with the function datatable of DT library , i create HTML widget to display get_num.
#In the following DT table, we can interactively visualize all numerical attributes and our response variable, credit worthiness (Class) . We can search / filter any keywords or specific entries  for each variable.

get_num <-  select_if(german, is.numeric)
get_num <- get_num %>% 
  mutate(Class = german$Class)
DT::datatable(get_num,rownames = FALSE, filter ="top", options =list(pageLength=10,scrollx=T))



# qualitative attributes : CheckingAccountStatus , CreditHistory, Purpose ,SavingsAccountBonds, etc ----
#################################################################################################

# CheckingAccountStatus

ggplot(german, aes(CheckingAccountStatus, ..count..)) + 
  geom_bar(aes(fill = Class), position = "dodge") +
  labs(title="bar plot", subtitle="AccountStatus grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# CreditHistory

german %>% 
  ggplot(aes(CreditHistory, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge") +
  labs(title="bar plot", subtitle="Credit history grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# Purpose

german %>% 
  ggplot(aes(Purpose)) +
  geom_bar(aes(fill=Class), width = 0.5) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) + 
  labs(title="Histogram", 
       subtitle="credit purpose across credit worthiness",
       caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


purp <- table(german$Purpose,german$Class)

ggballoonplot(as.data.frame(purp), fill = "value",title="Ballon plot",
              subtitle="credit purpose across credit worthiness",
              caption = "source: german credit data")+
  scale_fill_viridis_c(option = "C") 


# SavingsAccountBonds

german %>% 
  ggplot(aes(SavingsAccountBonds, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge") +
  labs(title="bar plot", subtitle="Saving accounts grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# EmploymentDuration

german %>% 
  ggplot(aes(EmploymentDuration, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge") +
  labs(title="bar plot", subtitle="Employment Duration grouped by credit worthiness",    caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# ForeignWorker

german %>% 
  ggplot(aes(ForeignWorker, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge") +
  labs(title="bar plot", subtitle="Foreignworker grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# Telephone

german %>% 
  ggplot(aes(Telephone, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge")+
  labs(title="bar plot", subtitle="Telephone grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))


# OtherInstallmentPlans

german %>% 
  ggplot(aes(OtherInstallmentPlans, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge")+
  labs(title="bar plot", subtitle="OtherInstallmentPlans grouped by credit worthiness",   caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))



# OtherDebtorsGuarantor

german %>% 
  ggplot(aes(OtherDebtorsGuarantors, ..count..)) + 
  geom_bar(aes(fill=Class), position ="dodge")+
  labs(title="bar plot", subtitle="OtherDebtorsGuarantors grouped by credit worthiness", caption = "source: german credit data") +
  scale_fill_manual(values = c("Good" = "darkblue", "Bad" = "red"))



# i create a dataframe "get_cat" which contains all categorical attributes including the response variable. Then, with the function datatable of DT library , i create HTML widget to display get_num.
#In the following DT table, we can interactively visualize all categorical features and our response variable, credit worthiness (Class) . We can search / filter any keywords or specific entries  for each variable.

get_cat <- select_if(german, is.factor)
DT::datatable(get_cat,rownames = FALSE, filter ="top", options =list(pageLength=10,scrollx=T))



# qualitative+ quantitative attributes : Multivariate analysis ----
############################################################


# Age vs credit amount for various purpose

mycol <- c("red","blue")

xyplot(Amount ~ Age|Purpose, german,
       grid = TRUE, 
       group= Class,
       auto.key = list(points = TRUE, rectangles = FALSE, title="Class",cex=0.7, space = "right"),
       main=list(
         label="Age vs credit amount for various purposes\n grouped by credit worthiness",
         cex=1),
       sub= "source: german credit data",
       par.settings = c(simpleTheme(col = mycol),list(par.sub.text = list(cex = 0.7, col = "black",x=0.75))))


# Age vs credit Amount for Personal status and sex

xyplot(Amount ~ Age |Personal , german,
       group = Class,
       grid = TRUE,
       auto.key = list(points = TRUE, rectangles = FALSE, title="Class",cex=0.7, space = "right"),
       main=list(
         label="Age vs credit amount for Personal status and sex\n grouped by credit worthiness",
         cex=1),
       #sub= "source: german credit data"),
       sub= "source: german credit data",
       par.settings = c(simpleTheme(col = mycol),list(par.sub.text = list(cex = 0.7, col = "black",x=0.75))))


# Distribution Amount ~ Age conditioned on Personal status/sex 

histogram(Amount ~ Age | Personal, 
          data = german, 
          xlab = "Age",
          ylab= "Amount",
          main= list( label="Distribution of Age and Personal status & sex", cex=1),
          col="purple",
          sub= "source: german credit data",
          par.settings=list(par.sub.text = list(cex = 0.7, col = "black", x=0.75)))



##III. ------------------------------- Section 3: Data Preprocessing  ----------------------------------------------------------


# 3.1. Data wrangling ---------------------------------------------------------------------------------


#To perform the different steps of data pre-processing part and successive analysis , 
#we make a copy of the german credit dataset, in order to  to keep unchanged our original german credit data .
german.copy <- german


# 3.1.1. Missing values ----

#we have no missing values in our data, compared to the german credit data downloaded from kaggle.
#the latter is only to purpose, to see how would be the aggr plot in presence of NA values.

missing.values <- aggr(german.copy, sortVars = T, prop = T, sortCombs = T,
                       cex.lab = 1.5, cex.axis = .6, cex.numbers = 5,
                       combined = F, gap = -.2)


#german credit data from kaggle
GCD <- read.csv(file="german_credit_data.csv", header=TRUE, sep=",")
germanKaggle_NA <- aggr(GCD, sortVars = T, prop = T, sortCombs = T,
                        cex.lab = 1.5, cex.axis = .6, cex.numbers = 5,
                        combined = F, gap = -.2)

# 3.1.2. renaming variables ----

german.copy = rename( german.copy,
                      Risk = `Class`,
                      installment_rate = `InstallmentRatePercentage`,
                      present_resid = `ResidenceDuration`,
                      n_credits = `NumberExistingCredits`,
                      n_people = `NumberPeopleMaintenance`,
                      check_acct = `CheckingAccountStatus`,
                      credit_hist = `CreditHistory`,
                      savings_acct = `SavingsAccountBonds`,
                      present_emp = `EmploymentDuration`, 
                      status_sex = `Personal`,
                      other_debt = `OtherDebtorsGuarantors`,
                      other_install = `OtherInstallmentPlans`
)

#get glimpse of data

glimpse(german.copy)



# 3.1.3. Skewness ----

#NB: Read carefully in the pdf and rmd reports, comments/lines about the necessity or not to perform a skewness section in our study.

mySkew <- preProcess(german.copy, method = c("BoxCox"))
mySkew
mySkew$method
mySkew$bc

#We visually inspected the skewness by plotting histogram of the 6 numerical attributes inclined to a BoxCox transformation ( see mySkew$method)

p1<- german.copy %>% 
  ggplot(aes(x=Amount)) + 
  geom_histogram(aes(y=..density..),    # Histogram with density 
                 binwidth=1000,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of Amount", subtitle= "overlaid with kernel density curve", caption = "source:german credit data")


p2<- german.copy %>% 
  ggplot(aes(x=installment_rate)) + 
  geom_histogram(aes(y=..density..),    # Histogram with density 
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of installment rate percentage", subtitle= "overlaid with kernel density curve", caption = "source:german credit data")

p3 <-  german.copy %>%
  ggplot(aes(x=Duration)) + 
  geom_histogram(aes(y=..density..), # Histogram with density instead y-axis count
                 binwidth=5,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of Duration in months", subtitle= "overlaid with kernel density curve", caption = "source:german credit data")


p4 <-  german.copy %>%
  ggplot(aes(x=n_credits)) + 
  geom_histogram(aes(y=..density..),      
                 binwidth=1,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of number of existing credits", subtitle= "overlaid with kernel density curve", caption = "source:german credit data")

p5 <-  german.copy %>%
  ggplot(aes(x=Age)) + 
  geom_histogram(aes(y=..density..), # Histogram with density instead of count on y-axis
                 binwidth=5,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of Age", subtitle= "overlaid with kernel density curve", caption = "source:german credit data")

p6 <-  german.copy %>%
  ggplot(aes(x=present_resid)) + 
  geom_histogram(aes(y=..density..),
                 binwidth=1,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of residence duration", subtitle= "overlaid with kernel density curve", caption = "source:german credit data")

#plot all 
grid.arrange(p1, p3, p2, p4, p5, p6, ncol=3)


#transformed data

myTransformed <- predict(mySkew, german.copy)
glimpse(myTransformed)

p1_new <- myTransformed %>% 
  ggplot(aes(x=Amount)) + 
  geom_histogram(aes(y=..density..),
                 binwidth=.5,
                 colour="black", fill="white") +
  geom_density(alpha=.5, fill="purple") +  # Overlay with transparent density plot
  labs(title= "Histogram of Amount", subtitle= "overlaid with kernel density curve", 
       caption = "source: my transformed german data with no Skewness")

p1_new 



#Before to continue with the Feature selection step, we recoded the response variable **Risk**, 
#by creating a new variable (numerical) **risk_bin** where **0** corresponds to a Bad credit worthiness and **1** to a Good credit worthiness.
#We didn't eliminate the original Risk outcome , they can be both useful for successive analysis depending of the Machine learning model we'll test.

german.copy <- german.copy %>%
  mutate(risk_bin = ifelse(Risk == "Bad",0,1))




# 3.2. Feature Selection ---------------------------------------------------------------------------------


# 3.2.1 . filter methods ----


#sbf (selection by filtering, caret package)

filterCtrl <- sbfControl(functions = rfSBF, method = "repeatedcv", repeats = 5)

set.seed(1)
rfWithFilter <- sbf( Risk ~ ., german.copy[,-22], sbfControl = filterCtrl)
rfWithFilter


#WOE,IV (Information package) 

IV <- create_infotables(data=german.copy[,-21], NULL, y="risk_bin", bins=10) 
IV$Summary$IV <- round(IV$Summary$IV*100,2)

#get tables from the IV statistic
IV$Tables

#get summary of the IV statistic
IV$Summary

#very very weak predictor (IV< 2%)
kable(IV$Summary %>%
        filter(IV < 2))
#very weak predictor (2%<=IV< 10%)
kable(IV$Summary %>%
        filter(IV >= 2 & IV < 10 ))
#medium prediction power (10%<=IV< 30%)
kable(IV$Summary %>%
        filter(IV >= 10 & IV < 30 ))
#good prediction power: but There is no strong predictor with $IV$ between 30% to 50%
kable(IV$Summary %>%
        filter(IV >= 30 & IV < 50 ))
#very high prediction power (IV > 50%)
kable(IV$Summary %>%
        filter( IV > 50 ))



# 3.2.2 . wrapper methods ----


#rfe (recursive feature elimination, caret package)

#We created a function called "**recur.feature.selection**" which internally defined :  x, the data frame of predictor variables ;
#y, the  vector (numeric or factor) of outcome (y) ;  sizes, an integer vector for the specific subset sizes that should be tested, and a list of options that can be used to specify the model and the methods for prediction(using rfecontrol). Then we applied the algorithm using the rfe function.

recur.feature.selection <- function(num.iters=20, features.var, outcome.var){
  set.seed(10)
  sizes.var <- 1:20
  control <- rfeControl(functions = rfFuncs, #pre-defined sets of functions,randomForest(rffuncs)
                        method = "cv",
                        number = num.iters,
                        returnResamp = "all",
                        verbose = FALSE
  )
  results.rfe <- rfe(x = features.var, 
                     y = outcome.var,
                     sizes = sizes.var,
                     rfeControl = control)
  return(results.rfe)
}


#To resolve the summary.connection(connection) : invalid connection, i install the doSnow-parallel-DeLuxe R script performed by Tobias Kind (2015); see more here : https://github.com/tobigithub/R-parallel/wiki/R-parallel-Errors

source("Install-doSNOW-parallel-DeLuxe.R") 

#i remove in the features.var (both Risk (21) and risk_bin (22))  /  i keep only Risk (factor) as outcome var.

rfe.results <- recur.feature.selection(features.var = german.copy[,c(-21,-22)],   
                                       outcome.var = german.copy[,21])    

# view results
rfe.results



#Boruta algorithm

set.seed(123)
Boruta.german <- Boruta(Risk ~ . , data = german.copy[,-22], doTrace = 0, ntree = 500)
Boruta.german

#Functions which convert the Boruta selection into a formula which returns only Confirmed attributes.
getConfirmedFormula(Boruta.german)

#plot boruta object
plot(Boruta.german,las=2, main="Boruta result plot for german data")


# 3.2.3. Embedded methods ----


#read text on pdf and rmd reports


#At the end, Based on the Weight of Evidence and Information Value statistics results, we will keep attributes which $IV > 2$. 
#More over, the wrapper methods , recursive feature elimination and Boruta algorithm confirm this choice since they helped to investigate better on the suspicious character of the highest power predictor, check_acct. The important attributes validated by these algorithms corroborate with our choice.

#attributes to keep (IV>2)
keep <- IV$Summary %>%
  filter( IV > 2)

#german.copy data with attributes(IV>2) and response variable
german.copy2 <- german.copy[,c(keep$Variable,"Risk")]

#for convenience, we change the levels  Good = 1,  Bad = 0
german.copy2$Risk <- as.factor(ifelse(german.copy2$Risk == "Bad", '0', '1'))

#get a glimpse of german copies , see the differences./ german.copy2 is our subsetted data after pre-processing

glimpse(german.copy2)




#3.3. Data partitioning -----------------------------------------------------------------------------

index <- createDataPartition(y = german.copy2$Risk, p = 0.7, list = F)

# i create a function to calculate percent distribution for factors
pct <- function(x){
  tbl <- table(x)
  tbl_pct <- cbind(tbl,round(prop.table(tbl)*100,2))
  colnames(tbl_pct) <- c('Count','Percentage')
  kable(tbl_pct)
}

#training set and test set
train <- german.copy2[index,]
pct(train$Risk)

test <- german.copy2[-index,]
pct(test$Risk)



##IV. ------------------------------- Section 4: Methods and Analysis  ----------------------------------------------------------

#All references and formulas are detailed in the rmd and pdf reports . 



##V. ------------------------------- Section 5: Results  ----------------------------------------------------------


#5.1. Identifying the best model ----


#5.1.1. Logistic Regression Models

#fit glm model on training set
set.seed(1)                               
reg_model <- glm(formula= Risk ~ . , data=train, family="binomial")

#summary of the results of fitted logistic regression model 
summary(reg_model)



#Calculation of variable importance for  glm model  with repeated 10 folds cross-validation ,
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model_reg <- train(Risk ~ ., data=train, method="glm",
                   trControl=control)

importance <- varImp(model_reg, scale=FALSE)
plot(importance)

#fit new glm model on training set
set.seed(1)
reg_model.new <- glm(formula= Risk ~ check_acct + Purpose + credit_hist + Amount + present_emp, data=train, family="binomial")
summary(reg_model.new)


#reg_model: predict on test set
reg_predictions <- predict(reg_model, test, type="response")
reg_predictions <- round(reg_predictions) %>% factor()
cm_logreg.m1<-confusionMatrix(data= reg_predictions, reference=test$Risk, positive='0')
cm_logreg.m1


#reg_model.new: predict on test set
reg_predictions.new <- predict(reg_model.new, test, type="response")
reg_predictions.new <- round(reg_predictions.new) %>% factor()
cm_logreg.m1_1 <- confusionMatrix(data=reg_predictions.new, reference=test$Risk, positive='0')
cm_logreg.m1_1


#Model performance plot

#define performance

reg_prediction.values <- predict(reg_model, test[,-16], type="response")
prediction_reg.model <- prediction(reg_prediction.values, test[,16])
reg_model.perf <- performance(prediction_reg.model,"tpr","fpr")

reg2_prediction.values <- predict(reg_model.new, test[,-16], type="response")
prediction_reg.modelnew <- prediction(reg2_prediction.values, test[,16])
reg2_model.perf <- performance(prediction_reg.modelnew,"tpr","fpr")


#plot Roc curve

#reg_model
plot(reg_model.perf, lwd=2, colorize=TRUE, main="ROC curve, m1: Logistic Regression Performance")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

#reg_model.new
plot(reg2_model.perf, lwd=2, colorize=TRUE, main="ROC curve, m1.1: Logistic Regression with selected variables")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)


# Plot precision/recall curve
reg_model.precision <- performance(prediction_reg.model, measure = "prec", x.measure = "rec")
plot(reg_model.precision, main="m1 :Precision/recall curve")
reg2_model.precision <- performance(prediction_reg.modelnew , measure = "prec", x.measure = "rec")
plot(reg2_model.precision, main="m1.1 :Precision/recall curve")


#AUC : logistic regression model
reg_model.AUROC <- round(performance(prediction_reg.model, measure = "auc")@y.values[[1]], 4)
reg_model.AUROC

reg2_model.AUROC <- round(performance(prediction_reg.modelnew, measure = "auc")@y.values[[1]], 4)
reg2_model.AUROC



#5.1.2. Decision trees

#fit decision tree model on training set using the rpart function
set.seed(1)
dt_model <- rpart(Risk~. , data=train, method ="class")

#we display CP table for Fitted Rpart Object
printcp(dt_model)

# Plot the rpart dt_model
prp(dt_model,type=2,extra=1, main="Tree:Recursive Partitioning")
#A wrapper for plotting rpart trees using prp
fancyRpartPlot(dt_model)

#predict on test set
set.seed(1)
dt_predictions <- predict(dt_model, test[,-16], type="class")
cm_dtree.m2 <- confusionMatrix(data=dt_predictions, reference=test[,16], positive="0")
cm_dtree.m2


#pruning tree
set.seed(1)
# prune by lowest cp
pruned.tree <- prune(dt_model, 
                     cp = dt_model$cptable[which.min(dt_model$cptable[,"xerror"]),"CP"])
length(pruned.tree$frame$var[pruned.tree$frame$var == "<leaf>"])

#Displays CP table for Fitted pruned tree
printcp(pruned.tree)

#count the number of splits
length(grep("^<leaf>$" , as.character(pruned.tree$frame$var))) - 1

# Plot the rpart dt_model
prp(pruned.tree, type = 1, extra = 3, split.font = 2, varlen = -10)
#A wrapper for plotting rpart pruned tree using prp
fancyRpartPlot(pruned.tree)

#prediction on test set
pruned_predictions <- predict(pruned.tree, test[,-16], type="class")
cm_dtree.m2_1 <- confusionMatrix(data=pruned_predictions, reference=test[,16], positive="0")
cm_dtree.m2_1


#Model performance plot

#define performance

# score test data/model scoring for ROc curve - dt_model
dt_predictions.values <- predict(dt_model,test,type="prob")
dt_pred <- prediction(dt_predictions.values[,2],test$Risk)
dt_perf <- performance(dt_pred,"tpr","fpr")

# score test data/model scoring for ROc curve - pruned.tree
dt2_predictions.values <- predict(pruned.tree,test,type="prob")
dt2_pred <- prediction(dt2_predictions.values[,2],test$Risk)
dt2_perf <- performance(dt2_pred,"tpr","fpr")


#plot Roc curve

#dt_model
plot(dt_perf, lwd=2, colorize=TRUE, main=" ROC curve : Decision tree performance")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)
#pruned.tree
plot(dt2_perf, lwd=2, colorize=TRUE, main="ROC curve : pruned tree performance")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)


# Plot precision/recall curve
dt_perf_precision <- performance(dt_pred, measure = "prec", x.measure = "rec")
plot(dt_perf_precision, main="Decision tree :Precision/recall curve")
dt2_perf_precision <- performance(dt2_pred, measure = "prec", x.measure = "rec")
plot(dt2_perf_precision, main="pruned tree :Precision/recall curve")


#AUC : Decision trees
dt_model.AUROC <- round(performance(dt_pred, measure = "auc")@y.values[[1]], 4)
dt_model.AUROC

dt2_model.AUROC <- round(performance(dt2_pred, measure = "auc")@y.values[[1]], 4)
dt2_model.AUROC


#5.1.3. random forests


#fit random forest model on training set using the randomForest function
set.seed(1)
rf_model <- randomForest(Risk ~ ., data = train,importance=TRUE)
# print random forest model
print(rf_model)


#prediction on test set
rf_predictions <- predict(rf_model, test[,-16], type="class")
cm_rf.m3 <- confusionMatrix(data=rf_predictions, reference=test[,16], positive="0")
cm_rf.m3


#plot variable importance
varImpPlot(rf_model, main="Random Forest: Variable Importance")


# Get importance
importance <- randomForest::importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables

ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables', title="random forest variable importance: MeanDecreaseGini") +
  coord_flip() + 
  theme_few()


#fit new rf model
set.seed(1)
rf_model.new <- randomForest(Risk ~ Amount + check_acct + Age + Purpose + Duration  ,
                             data = train,
                             importance=TRUE
)

#prediction on test set 
rf_predictions.new <- predict(rf_model.new, test[,-16], type="class")
cm_rf.m3_1 <- confusionMatrix(data=rf_predictions.new , reference=test[,16], positive="0")
cm_rf.m3_1


#find the best parameters  with 10-fold cross validation  to build optimal random forest model
set.seed(1)
nodesize.vals <- c(2, 3, 4, 5)
ntree.vals <- c(200, 500, 1000, 2000)
tuning.results <- tune.randomForest(Risk ~ ., data = train, mtry=3, nodesize=nodesize.vals, ntree=ntree.vals)
print(tuning.results)


#prediction on test set
rf_model.best <- tuning.results$best.model
rf_predictions.best <- predict(rf_model.best, test[,-16], type="class")
cm_rf.m3_2 <-confusionMatrix(data=rf_predictions.best, reference=test[,16], positive="0")
cm_rf.m3_2



#Model performance plot

#define performance

# score test data/model scoring for ROc curve - rf_model
rf_predictions.values <- predict(rf_model,test,type="prob")
rf_pred <- prediction(rf_predictions.values[,2],test$Risk)
rf_perf <- performance(rf_pred,"tpr","fpr") 

# score test data/model scoring for ROc curve - rf_model.new
model.new_predictions.values <- predict(rf_model.new,test,type="prob")
model.new_pred <- prediction(model.new_predictions.values[,2],test$Risk)
rf.new_perf <- performance(model.new_pred,"tpr","fpr") 

# score test data/model scoring for ROc curve - rf_model.best
model.best_predictions.values <- predict(rf_model.best,test,type="prob")
model.best_pred <- prediction(model.best_predictions.values[,2],test$Risk)
rf.best_perf <- performance(model.best_pred,"tpr","fpr") 


#plot roc curves

#rf_model
plot(rf_perf, lwd=2, colorize=TRUE, main=" ROC curve : random forest performance ")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4) 

#rf_model.best
plot(rf.best_perf, lwd=2, colorize=TRUE, main="ROC curve: random forest - tuning parameters")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4) 


# Plot precision/recall curve

rf_perf_precision <- performance(rf_pred, measure = "prec", x.measure = "rec")
plot(rf_perf_precision , main="rf_model: Plot Precision/recall curve")

rf.best_perf_precision <- performance(model.best_pred, measure = "prec", x.measure = "rec")
plot(rf.best_perf_precision, main="rf_model.best: Plot Precision/recall curve")



#AUC : random forests models
rf_model.AUROC <- round(performance(rf_pred, measure = "auc")@y.values[[1]], 4)
rf_model.AUROC

rf_newmodel.AUROC <- round(performance(model.new_pred, measure = "auc")@y.values[[1]], 4)
rf_newmodel.AUROC

rf_bestmodel.AUROC <- round(performance(model.best_pred, measure = "auc")@y.values[[1]], 4)
rf_bestmodel.AUROC




#5.1.4. Support Vector Machine


#we fit a SVM model with radial basis kernel and cost as 100
svm_model <- svm(Risk ~. , data = train, kernel = "radial", gamma = 1, cost = 100)
print(svm_model)


#prediction on test set
set.seed(1)
svm_predictions <- predict(svm_model, test[,-16], type="class")
cm_svm.m4 <- confusionMatrix(data=svm_predictions, reference=test[,16], positive="0")
cm_svm.m4


#important features - selected by 10 fold cv , method = svmRadial
set.seed(1)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model_svm <- train(Risk ~ ., data=train, method="svmRadial",
                   trControl=control)

importance_svm <- varImp(model_svm,scale=FALSE) 
plot(importance_svm,cex.lab=0.5)


#tuning parameters svm model

set.seed(1)
cost.weights<- c(0.1,10,100)
gamma.weights <- c(0.01,0.25,0.5,1)

tuning.results <- tune(svm,Risk ~ check_acct + credit_hist +  Duration + savings_acct + Age, data =train, kernel ="radial",  ranges = list(cost=cost.weights, gamma = gamma.weights))

print(tuning.results)

plot(tuning.results)


#prediction on unseen data , test set
svm_model.best <- tuning.results$best.model
svm_predictions.best <- predict(svm_model.best, test[,-16])
cm_svm.m4_1 <- confusionMatrix(data=svm_predictions.best, reference=test[,16], positive="0")
cm_svm.m4_1


#fitted svm model based on vanilladot Linear kernel function
set.seed(1)
svm_model.vanilla <- ksvm(Risk ~., data = train, kernel = "vanilladot")
print(svm_model.vanilla)

#prediction on test set
svm_predictions.vanilla <- predict(svm_model.vanilla, test[,-16])
cm_svm.m4_2 <- confusionMatrix(data=svm_predictions.vanilla, reference=test[,16], positive="0")
cm_svm.m4_2



#Model performance plot

#define performance

# score test data/model scoring for ROc curve - svm_model
svm_predictions.values <- predict(svm_model,test,type="decision", decision.values =TRUE)
svm_pred <- prediction(attributes(svm_predictions.values)$decision.values,test$Risk)
svm_perf <- performance(svm_pred,"tpr","fpr") 


# score test data/model scoring for ROc curve - svm_model.best
svm.best_predictions.values <- predict(svm_model.best,test,type="decision", decision.values =TRUE)
svm.best_pred <- prediction(attributes(svm.best_predictions.values)$decision.values,test$Risk)
svm.best_perf <- performance(svm.best_pred,"tpr","fpr")


# score test data/model scoring for ROc curve - svm_model.vanilla
svm.van_predictions.values <- predict(svm_model.vanilla,test,type="decision")
svm.van_pred <- prediction(svm.van_predictions.values,test$Risk)
svm.van_perf <- performance(svm.van_pred,"tpr","fpr")



#plot roc curves

#svm_model.best
plot(svm.best_perf, lwd=2, colorize=TRUE, main="SVM: ROC curve - radial basis,important features")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4) 

#svm_model.vanilla
plot(svm.van_perf , lwd=2, colorize=TRUE, main="SVM: ROC curve - vanilladot")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4) 


# Plot precision/recall curve
svm.best_perf_precision <- performance(svm.best_pred, measure = "prec", x.measure = "rec")
plot(svm.best_perf_precision, main="svm_model.best: Plot Precision/recall curve")

svm.van_perf_precision <- performance(svm.van_pred, measure = "prec", x.measure = "rec")
plot(svm.van_perf_precision, main="svm_model.van: Plot Precision/recall curve")


#AUC : svm models
svm_model.AUROC <- round(performance(svm_pred, measure = "auc")@y.values[[1]], 4)
svm_model.AUROC

svm.best_model.AUROC <- round(performance(svm.best_pred, measure = "auc")@y.values[[1]], 4)
svm.best_model.AUROC

svm.van_model.AUROC <- round(performance(svm.van_pred, measure = "auc")@y.values[[1]], 4)
svm.van_model.AUROC




#5.1.5. Neural Network


# Data pre-processing for Neural network models


#retain from the german.copy data only the numerical target "risk_bin" and "keep$Variables" from the IV statistic.

german.copy3 <- german.copy[,c(keep$Variable,"risk_bin")]


# Read a numeric copy: Numeric data for Neural network 
german.copy3 <- as.data.frame(sapply(german.copy3, as.numeric ))


# For neural network we would need continuous data
# Sampling for Neural Network - It can be used for Lasso regression too
set.seed(1)
idx <- createDataPartition(y = german.copy3[,16], p = 0.7, list = F)

# Training Sample for Neural Network
train_num <- german.copy3[idx,] # 70% here

# Test Sample for Neural Network
test_num <- german.copy3[-idx,] # rest of the 30% data goes here


#structures of new train/test sets

str(train_num)
str(test_num)


# Normalization / scaling
train_num$risk_bin <- as.factor(train_num$risk_bin)
test_num$risk_bin <- as.factor(test_num$risk_bin)

# Function: Normalize using Range

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

train_num_norm <- as.data.frame(lapply(train_num[,1:15], normalize ))
test_num_norm <- as.data.frame(lapply(test_num[,1:15], normalize ))


train_num_norm$risk_bin <- as.factor(ifelse(train_num$risk_bin == 1, 1, 0))
test_num_norm$risk_bin <- as.factor(ifelse(test_num$risk_bin == 1, 1, 0))


# train_num_norm <- as.data.frame(lapply(train_num[,1:24], scale )) # use scale if normal
# test_num_norm <- as.data.frame(lapply(test_num[,1:24], scale ))   # use scale if normal


# build the neural network (NN) formula
a <- colnames(train_num[,1:15])
mformula <- as.formula(paste('risk_bin ~ ' , paste(a,collapse='+')))

set.seed(1234567890)
train_nn <- train_num_norm
test_nn <- test_num_norm


# Neural network modelling
set.seed(1)
neur.net <- nnet(risk_bin~., data=train_nn,size=20,maxit=10000,decay=.001, linout=F, trace = F)
print(neur.net)


#prediction of neur.net on test set
neur.net_predictions <- predict(neur.net,newdata=test_nn, type="class") %>% factor()
cm_neuralnet.m5 <- confusionMatrix(data=neur.net_predictions, reference=test_nn[,16], positive="0")
cm_neuralnet.m5

# get the weights and structure in the right format
wts <- neuralweights(neur.net)
struct <- wts$struct
wts <- unlist(wts$wts)
#plot
plotnet(wts, struct=struct)


#Model performance plot

#define performance

# score test data/model scoring for ROc curve - neural network model
neur.net_pred <- prediction(predict(neur.net, newdata=test_nn, type="raw"),test_nn$risk_bin)
neur.net_perf <- performance(neur.net_pred,"tpr","fpr")

#plot roc curves
plot(neur.net_perf ,lwd=2, colorize=TRUE, main="ROC curve:  Neural Network performance")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)  

# Plot precision/recall curve
neur.net_perf_precision <- performance(neur.net_pred, measure = "prec", x.measure = "rec")
plot(neur.net_perf_precision, main="neural network: Plot Precision/recall curve")

#AUC : neural network model
neur_model.AUROC <- round(performance(neur.net_pred, measure = "auc")@y.values[[1]], 4)
neur_model.AUROC



#5.1.6. Lasso regression

#For the lasso regression model, we will use the same train_num and test_num we defined first for the neural network model. However, because the lars and glmnet functions don't need a regression formula , but input matrices of predictors,
#we will first construct  *mat_train* and *mat_test* with the model.matrix function as follow.

mat_train <- model.matrix(risk_bin ~ . , data = train_num  ) # convert to numeric matrix
mat_test <- model.matrix(risk_bin ~ . , data = test_num  )  # convert to numeric matrix


#fit LASSO model
set.seed(1)
lasso_model <- cv.glmnet(mat_train,as.numeric(train_num$risk_bin), alpha=1, nfolds=10, family="binomial", type.measure = 'auc')
print(lasso_model)
#plot
plot(lasso_model)


#prediction on test set
lasso_predictions <- predict(lasso_model, newx = mat_test, s = "lambda.1se", type = "response") 
lasso_predictions <- as.factor(ifelse(lasso_predictions>0.5,1,0))
cm_lasso.m6 <- confusionMatrix(data=lasso_predictions, reference=test_num$risk_bin, positive='0')
cm_lasso.m6


#Model performance plot

#define performance

# score test data/model scoring for ROc curve - lasso_model
lasso.prob <- predict(lasso_model,type="response", newx =mat_test, s = 'lambda.1se')
lasso_model_pred <- prediction(lasso.prob,test_num$risk_bin)
lasso_model_perf <- performance(lasso_model_pred,"tpr","fpr")

#plot roc curve
#lasso_model
plot(lasso_model_perf,colorize=TRUE,  main="LASSO: ROC curve - type measure:auc") 
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

# Plot precision/recall curve
lasso_perf_precision <- performance(lasso_model_pred, measure = "prec", x.measure = "rec")
plot(lasso_perf_precision, main="lasso_model: Plot Precision/recall curve")

#AUC : Lasso model
lasso_model.AUROC <- round(performance(lasso_model_pred, measure = "auc")@y.values[[1]], 4)
lasso_model.AUROC



#5.2. Model optimization ----

#5.2.1. Model Performance Comparision - Measures of Accuracy 

#Balanced Accuracy , F1-score, Recall, Precision


# models Performance Table
models <- c('m1:Logistic regression', 'm1_1:Logistic regression - important vars',
            'm2:Decision tree','m2_1:Decision tree - pruning', 
            'm3:Random forest', 'm3_1: Random forest - important vars',
            'm3_2: Random forest- tuning parameters',
            "m4:SVM,radial", "m4_1:SVM,radial - important vars",  
            "m4_2:SVM,vanilladot", "m5: Neural net", "m6:Lasso reg")


#i get overall accuracy for each model 
avg_acc <- round(c( cm_logreg.m1$overall[['Accuracy']],
                    cm_logreg.m1_1$overall[['Accuracy']],
                    cm_dtree.m2$overall[['Accuracy']],
                    cm_dtree.m2_1$overall[['Accuracy']],
                    cm_rf.m3$overall[['Accuracy']],
                    cm_rf.m3_1$overall[['Accuracy']],
                    cm_rf.m3_2$overall[['Accuracy']],
                    cm_svm.m4$overall[['Accuracy']],
                    cm_svm.m4_1$overall[['Accuracy']],
                    cm_svm.m4_2$overall[['Accuracy']],
                    cm_neuralnet.m5$overall[['Accuracy']],
                    cm_lasso.m6$overall[['Accuracy']]),4 )

#balanced accuracy
balanced_acc <- round(c(cm_logreg.m1$byClass[['Balanced Accuracy']],
                        cm_logreg.m1_1$byClass[['Balanced Accuracy']],
                        cm_dtree.m2$byClass[['Balanced Accuracy']],
                        cm_dtree.m2_1$byClass[['Balanced Accuracy']],
                        cm_rf.m3$byClass[['Balanced Accuracy']],
                        cm_rf.m3_1$byClass[['Balanced Accuracy']],
                        cm_rf.m3_2$byClass[['Balanced Accuracy']],
                        cm_svm.m4$byClass[['Balanced Accuracy']],
                        cm_svm.m4_1$byClass[['Balanced Accuracy']],
                        cm_svm.m4_2$byClass[['Balanced Accuracy']],
                        cm_neuralnet.m5$byClass[['Balanced Accuracy']],
                        cm_lasso.m6$byClass[['Balanced Accuracy']]),4)


#F1score
F1score.m1 <- round(F_meas(data=reg_predictions,reference=test$Risk),4)

F1score.m1_1 <- round(F_meas(data=reg_predictions.new,reference=test$Risk),4)

F1score.m2 <- round(F_meas(data=dt_predictions,reference=test$Risk),4)

F1score.m2_1 <- round(F_meas(data=pruned_predictions,reference=test$Risk),4)

F1score.m3 <- round(F_meas(data=rf_predictions, reference=test$Risk),4)
#2*(0.6734694*0.3666667) / ( 0.3666667 + 0.6734694 )

F1score.m3_1 <- round(F_meas(data=rf_predictions.new, reference=test$Risk),4)

F1score.m3_2 <- round(F_meas(data=rf_predictions.best, reference=test$Risk),4)

F1score.m4 <- round(F_meas(data=svm_predictions, reference=test$Risk),4)

F1score.m4_1 <- round(F_meas(data=svm_predictions.best , reference=test$Risk),4)

F1score.m4_2 <- round(F_meas(data=svm_predictions.vanilla, reference=test$Risk),4)  

F1score.m5 <- round(F_meas(data=neur.net_predictions , reference=test_nn$risk_bin),4)

F1score.m6 <- round(F_meas(data=lasso_predictions , reference=test_num$risk_bin),4)   

F1_score <- c(F1score.m1, F1score.m1_1,
              F1score.m2, F1score.m2_1,
              F1score.m3, F1score.m3_1,
              F1score.m3_2, F1score.m4,
              F1score.m4_1, F1score.m4_2,
              F1score.m5, F1score.m6)


#recall
recall <- round(c(cm_logreg.m1$byClass[['Sensitivity']],
                  cm_logreg.m1_1$byClass[['Sensitivity']],
                  cm_dtree.m2$byClass[['Sensitivity']],
                  cm_dtree.m2_1$byClass[['Sensitivity']],
                  cm_rf.m3$byClass[['Sensitivity']],
                  cm_rf.m3_1$byClass[['Sensitivity']],
                  cm_rf.m3_2$byClass[['Sensitivity']],
                  cm_svm.m4$byClass[['Sensitivity']],
                  cm_svm.m4_1$byClass[['Sensitivity']],
                  cm_svm.m4_2$byClass[['Sensitivity']],
                  cm_neuralnet.m5$byClass[['Sensitivity']],
                  cm_lasso.m6$byClass[['Sensitivity']]),4)


#precision
precision <- round(c(cm_logreg.m1$byClass[['Pos Pred Value']],
                     cm_logreg.m1_1$byClass[['Pos Pred Value']],
                     cm_dtree.m2$byClass[['Pos Pred Value']],
                     cm_dtree.m2_1$byClass[['Pos Pred Value']],
                     cm_rf.m3$byClass[['Pos Pred Value']],
                     cm_rf.m3_1$byClass[['Pos Pred Value']],
                     cm_rf.m3_2$byClass[['Pos Pred Value']],
                     cm_svm.m4$byClass[['Pos Pred Value']],
                     cm_svm.m4_1$byClass[['Pos Pred Value']],
                     cm_svm.m4_2$byClass[['Pos Pred Value']],
                     cm_neuralnet.m5$byClass[['Pos Pred Value']],
                     cm_lasso.m6$byClass[['Pos Pred Value']]),4)


# Combine Recall, Precision, balanced_acc, F1_score
model_performance_metric_1 <- as.data.frame(cbind(models, recall, precision, balanced_acc , F1_score, avg_acc))

# Colnames 
colnames(model_performance_metric_1) <- c("Models", "Recall", "Precision", "Balanced Accuracy", "F1-score", "Accuracy")

# kable performance table  metrics group1
kable(model_performance_metric_1,caption ="Comparision of Model Performances") %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ,color ="white", background = "blue") %>%
  column_spec(2,bold = T , color ="white", background = "lightsalmon") %>%
  column_spec(3,bold = T , color ="white", background = "darksalmon") %>%
  column_spec(4,bold = T ,color ="white", background = "coral" ) %>%
  column_spec(5,bold = T , color ="white", background = "tomato") %>%
  column_spec(6,bold =T ,color = "white" , background ="#D7261E")



#AUC, KS, GINI


#selected models
models2 <-  c('m1:Logistic regression',
              'm2:Decision tree',
              'm3_2: Random forest- tuning parameters',
              "m4_2:SVM,vanilladot", 
              "m5: Neural net",
              "m6:Lasso reg")

#i store all AUC values in models_AUC
models_AUC <- c (reg_model.AUROC,
                 dt_model.AUROC,
                 rf_bestmodel.AUROC,
                 svm.van_model.AUROC,
                 neur_model.AUROC,
                 lasso_model.AUROC)



#i calculate KS-test for each selected-fitted model

#logistic reg
m1.KS <- round(max(attr(reg_model.perf,'y.values')[[1]]-attr(reg_model.perf,'x.values')[[1]])*1, 4)

#decision tree
m2.KS <- round(max(attr(dt_perf,'y.values')[[1]]-attr(dt_perf,'x.values')[[1]])*1, 4)

# random forest 
m3_2.KS <- round(max(attr( rf.best_perf,'y.values')[[1]]-attr( rf.best_perf,'x.values')[[1]])*1, 4)

#SVM
m4_2.KS <- round(max(attr(svm.van_perf,'y.values')[[1]]-attr( svm.van_perf,'x.values')[[1]])*1, 4)

#Neural network
m5.KS <- round(max(attr(neur.net_perf,'y.values')[[1]]-attr( neur.net_perf,'x.values')[[1]])*1, 4)

#Lasso regression
m6.KS <- round(max(attr(lasso_model_perf,'y.values')[[1]]-attr(lasso_model_perf,'x.values')[[1]])*1, 4)

#i store all KS values in models_KS
models_KS <- c(m1.KS,
               m2.KS,
               m3_2.KS, 
               m4_2.KS,
               m5.KS,
               m6.KS)



#Gini : i calculate GINI values for each fitted model , 2*AUROC - 1

# Log. reg
m1.Gini <- (2 * reg_model.AUROC - 1)

# decision tree
m2.Gini <- (2 * dt_model.AUROC - 1)

# random forests
m3_2.Gini <- (2*rf_bestmodel.AUROC - 1)

# SVM 
m4_2.Gini <- (2*svm.van_model.AUROC- 1)

#Neural net
m5.Gini <- (2*neur_model.AUROC -1)

#Lasso
m6.Gini <- (2*lasso_model.AUROC -1 )

# i store all GINI values in models_Gini
models_Gini <- c(m1.Gini,
                 m2.Gini,
                 m3_2.Gini,
                 m4_2.Gini,
                 m5.Gini,
                 m6.Gini)

# Combine AUC, KS, GINI
model_performance_metric_2 <- as.data.frame(cbind(models2, models_AUC, models_KS, models_Gini))

# Colnames 
colnames(model_performance_metric_2) <- c("Models", "AUC", "KS", "Gini")


#kable performance table  metrics group2

kable(model_performance_metric_2,caption ="Comparision of Model Performances") %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  kable_styling(bootstrap_options = "bordered", full_width = F , position ="center") %>%
  column_spec(1,bold = T ,color ="white", background = "blue" ) %>%
  column_spec(2,bold = T ,color ="white", background = "coral" ) %>%
  column_spec(3,bold = T , color ="white", background = "tomato") %>%
  column_spec(4,bold =T ,color = "white" , background ="#D7261E")



#**So, why we prefer Lasso than logistic?** : Read text on pdf and rmd reports
lambda_1se <- lasso_model$lambda.1se
coef(lasso_model, s=lambda_1se)



#5.2.2. Model Performance Comparison - ROC curves


#Compare ROC Performance of Models
plot(reg_model.perf, col='blue', lty=1, lwd=2, 
     main='ROC curves: Model Performance Comparison') # logistic regression
plot(dt_perf, col='green',lty=2, add=TRUE,lwd=2); #simple decision tree 
plot(rf.best_perf, col='red',add=TRUE,lty=3,lwd=2.5); # random forest - tuning parameters
plot(svm.van_perf, col='Navy',add=TRUE,lty=4,lwd=2); # SVM, vanilladot
plot(neur.net_perf, col='gold',add=TRUE,lty=5,lwd=2); # Neural Network
plot(lasso_model_perf, col='purple',add=TRUE,lty=6,lwd=2); # Lasso regression
legend(0.55,0.4,
       c('m1:Logistic regression',
         'm2:Decision tree',
         'm3_2:Random forest- tuning parameters',
         "m4_2:SVM,vanilladot", 
         "m5: Neural net", 
         "m6:Lasso reg"),
       col=c("blue","green", "red","Navy", "gold", "purple"),
       lwd=3,cex=0.7,text.font = 4);
lines(c(0,1),c(0,1),col = "Black", lty = 4 ) # random line



##VI. ------------------------------- Section 6: Conclusions and suggestions  ----------------------------------------------------------

#text on pdf and rmd file



####################################THANKS TO READ THIS RSCRIPT AND EDIT IF NECESSARY #################################################################################



