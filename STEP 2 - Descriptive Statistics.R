### Importing of required libraries for the visualization and interpretation.
library(dplyr)
library(shiny)
library(tidyr)
library(tidyverse)
library(leaps)
library(knitr)
library(ggplot2)
library("reshape2")
library(caret)
library(class)
library(psych)
library(tree)
library(rpart)
library(rattle)
library(randomForest)
library(readxl)
library(moments)
library(FactoMineR)
library(glmnet)
library(corrplot)
library(rsample)
library(psych)
library("gridExtra")




#Bringing in the cleaned dataset
cervical_cancer <- read_csv("/Users/jasonrayen/Downloads/Jason Masters/Sem 2/Applications of Metadata in Complex Big Data Problems (AIT 582)/Project/cleaned_cervicalcancer.csv")

#STATISTICAL ANALYSIS

#Checking the structure of the data
str(cervical_cancer)

#Checking for Missing Values
table(is.na(cervical_cancer)) #No Null Values (Dataset cleaned in Python)

#Getting Summary Statistics of the dataset
summary(cervical_cancer) #The same will be cross checked post simulation to check if there are any major changes

#Data Simulation - Using the mean and standard deviation
set.seed(3000)

N = 3000 
simulated_cervicalCancer = data.frame(
  Age = round(rnorm(N,mean(cervical_cancer$Age),sd(cervical_cancer$Age))),
  `Number of sexual partners` = round(rnorm(N,mean(cervical_cancer$`Number of sexual partners`),sd(cervical_cancer$`Number of sexual partners`))),
  `First sexual intercourse` = round(rnorm(N,mean(cervical_cancer$`First sexual intercourse`),sd(cervical_cancer$`First sexual intercourse`))),
  `Num of pregnancies` = round(rnorm(N,mean(cervical_cancer$`Num of pregnancies`),sd(cervical_cancer$`Num of pregnancies`))),
  Smokes = round(rnorm(N,mean(cervical_cancer$Smokes),sd(cervical_cancer$Smokes))),
  `Smokes (years)` = round(rnorm(N,mean(cervical_cancer$`Smokes (years)`),sd(cervical_cancer$`Smokes (years)`))),
  `Smokes (packs/year)` = round(rnorm(N,mean(cervical_cancer$`Smokes (packs/year)`),sd(cervical_cancer$`Smokes (packs/year)`))),
  `Hormonal Contraceptives` = round(rnorm(N,mean(cervical_cancer$`Hormonal Contraceptives`),sd(cervical_cancer$`Hormonal Contraceptives`))),
  `Hormonal Contraceptives (years)` = round(rnorm(N,mean(cervical_cancer$`Hormonal Contraceptives (years)`),sd(cervical_cancer$`Hormonal Contraceptives (years)`))),
  IUD = round(rnorm(N,mean(cervical_cancer$IUD),sd(cervical_cancer$IUD))),
  `IUD (years)` = round(rnorm(N,mean(cervical_cancer$`IUD (years)`),sd(cervical_cancer$`IUD (years)`))),
  STDs = round(rnorm(N,mean(cervical_cancer$STDs),sd(cervical_cancer$STDs))),
  `STDs (number)` = round(rnorm(N,mean(cervical_cancer$`STDs (number)`),sd(cervical_cancer$`STDs (number)`))),
  `STDs:condylomatosis` = round(rnorm(N,mean(cervical_cancer$`STDs:condylomatosis`),sd(cervical_cancer$`STDs:condylomatosis`))),
  `STDs:cervical condylomatosis` = round(rnorm(N,mean(cervical_cancer$`STDs:cervical condylomatosis`),sd(cervical_cancer$`STDs:cervical condylomatosis`))),
  `STDs:vaginal condylomatosis` = round(rnorm(N,mean(cervical_cancer$`STDs:vaginal condylomatosis`),sd(cervical_cancer$`STDs:vaginal condylomatosis`))),
  `STDs:vulvo-perineal condylomatosis` = round(rnorm(N,mean(cervical_cancer$`STDs:vulvo-perineal condylomatosis`),sd(cervical_cancer$`STDs:vulvo-perineal condylomatosis`))),
  `STDs:syphilis` = round(rnorm(N,mean(cervical_cancer$`STDs:syphilis`),sd(cervical_cancer$`STDs:syphilis`))),
  `STDs:pelvic inflammatory disease` = round(rnorm(N,mean(cervical_cancer$`STDs:pelvic inflammatory disease`),sd(cervical_cancer$`STDs:pelvic inflammatory disease`))),
  `STDs:genital herpes` = round(rnorm(N,mean(cervical_cancer$`STDs:genital herpes`),sd(cervical_cancer$`STDs:genital herpes`))),
  `STDs:molluscum contagiosum` = round(rnorm(N,mean(cervical_cancer$`STDs:molluscum contagiosum`),sd(cervical_cancer$`STDs:molluscum contagiosum`))),
  `STDs:AIDS` = round(rnorm(N,mean(cervical_cancer$`STDs:AIDS`),sd(cervical_cancer$`STDs:AIDS`))),
  `STDs:HIV` = round(rnorm(N,mean(cervical_cancer$`STDs:HIV`),sd(cervical_cancer$`STDs:HIV`))),
  `STDs:Hepatitis B` = round(rnorm(N,mean(cervical_cancer$`STDs:Hepatitis B`),sd(cervical_cancer$`STDs:Hepatitis B`))),
  `STDs:HPV` = round(rnorm(N,mean(cervical_cancer$`STDs:HPV`),sd(cervical_cancer$`STDs:HPV`))),
  `STDs: Number of diagnosis` = round(rnorm(N,mean(cervical_cancer$`STDs: Number of diagnosis`),sd(cervical_cancer$`STDs: Number of diagnosis`))),
  `Dx:Cancer` = round(rnorm(N,mean(cervical_cancer$`Dx:Cancer`),sd(cervical_cancer$`Dx:Cancer`))),
  `Dx:CIN` = round(rnorm(N,mean(cervical_cancer$`Dx:CIN`),sd(cervical_cancer$`Dx:CIN`))),
  `Dx:HPV` = round(rnorm(N,mean(cervical_cancer$`Dx:HPV`),sd(cervical_cancer$`Dx:HPV`))),
  Dx = round(rnorm(N,mean(cervical_cancer$Dx),sd(cervical_cancer$Dx))),
  Hinselmann = round(rnorm(N,mean(cervical_cancer$Hinselmann),sd(cervical_cancer$Hinselmann))),
  Schiller = round(rnorm(N,mean(cervical_cancer$Schiller),sd(cervical_cancer$Schiller))),
  Citology = round(rnorm(N,mean(cervical_cancer$Citology),sd(cervical_cancer$Citology))),
  Biopsy = round(rnorm(N,mean(cervical_cancer$Biopsy),sd(cervical_cancer$Biopsy)))
)

#Checking Summary of simulated data
summary(simulated_cervicalCancer)

#Making all the values in the data consistent

table(cervical_cancer$Age)

table(cervical_cancer$`Number of sexual partners`)

table(cervical_cancer$`First sexual intercourse`)

table(cervical_cancer$`Num of pregnancies`)

table(cervical_cancer$Smokes)

table(cervical_cancer$`Smokes (years)`)
cervical_cancer$`Smokes (years)` <- round(cervical_cancer$`Smokes (years)`)

table(cervical_cancer$`Smokes (packs/year)`)
cervical_cancer$`Smokes (packs/year)` <- round(cervical_cancer$`Smokes (packs/year)`)

table(cervical_cancer$`Hormonal Contraceptives`)

table(cervical_cancer$`Hormonal Contraceptives (years)`)
cervical_cancer$`Hormonal Contraceptives (years)` <- round(cervical_cancer$`Hormonal Contraceptives (years)`)

table(cervical_cancer$IUD)

table(cervical_cancer$`IUD (years)`)
cervical_cancer$`IUD (years)` <- round(cervical_cancer$`IUD (years)`)

table(cervical_cancer$STDs)

table(cervical_cancer$`STDs (number)`)

table(cervical_cancer$`STDs:condylomatosis`)

table(cervical_cancer$`STDs:cervical condylomatosis`)

table(cervical_cancer$`STDs:vaginal condylomatosis`)



table(cervical_cancer$`STDs:vulvo-perineal condylomatosis`)

table(cervical_cancer$`STDs:syphilis`)

table(cervical_cancer$`STDs:pelvic inflammatory disease`)

table(cervical_cancer$`STDs:genital herpes`)

table(cervical_cancer$`STDs:molluscum contagiosum`)

table(cervical_cancer$`STDs:AIDS`)

table(cervical_cancer$`STDs:HIV`)

table(cervical_cancer$`STDs:Hepatitis B`)

table(cervical_cancer$`STDs:HPV`)

table(cervical_cancer$`STDs: Number of diagnosis`)

table(cervical_cancer$`Dx:Cancer`)

table(cervical_cancer$`Dx:CIN`)

table(cervical_cancer$`Dx:HPV`)

table(cervical_cancer$`Dx`)

table(cervical_cancer$Hinselmann)

table(cervical_cancer$Schiller)

table(cervical_cancer$Citology)

table(cervical_cancer$Biopsy)

## Producing some numerical and graphical summaries of the data set.

### Statistical Analysis

## summary(cervical_cancer) #summary of the dataset

# By observing the summary statistics of the cervical cancer dataset:

### Age: The patients are between the ages of 13 and 84, with a mean age of about 27.23.

### Sexual Behavior: There is a range of 1 to 28 sexual partners, with a median of 2. With a median age of 17, 
###   first sexual encounters usually happened between the ages of 10 and 32.

### Pregnancies: There were between 0 and 11 pregnancies overall, with an average of roughly 2.194 pregnancies per patient.

### Smoking: Approximately 14.31% of patients admitted being smokers. The mean duration of smoking is approximately 1.232 years, 
###.  while a few individuals report smoking for as long as 37 years. The stated number of packs smoked annually is roughly 0.4639 on average.

### Hormonal Contraceptives: The average patient's use of hormonal contraceptives was 2.214 years, accounting for 64.17% of cases.

### IUD: Intrauterine devices (IUDs) were used by approximately 11.25% of patients, with an average usage period of 0.5167 years.

### Sexually transmitted illnesses (STDs): 9.44% of patients reported having a history of these infections, with an average of 0.1611 STDs recorded.

### Cancer Indicators: A minor portion of patients reported symptoms similar to Hinselmann (4.58%), Schiller (9.58%), 
###.                   Citology (5.56%) and Biopsy (6.94%), which are conditions linked to cervical cancer.

### Other STDs: The prevalence of a number of other specific sexually transmitted diseases was comparatively low, 
###             typically less than 5%, including genital herpes, HPV, syphilis, condylomatosis and others.

## From these statistics:

### Smoking: A small percentage of patients reported they smoked.

### Contraception and STDs: While a smaller percentage reported having STDs, a considerable amount utilized hormonal contraception.

### Cancer Indicators: Some patients showed signs associated with cervical cancer,
###                    however these signs were not common in most of the individuals.

### These findings suggest that the dataset includes data on sexual behavior, medical history, demography and factors associated with cervical cancer. 
### The patient's stated symptoms and activities vary and some of them include risk factors linked to cervical cancer. To make explicit conclusions 
### about the associations between these characteristics and outcomes or incidence of cervical cancer, more thorough analysis and modelling is recquired.
 
#Data Simulation using normalization method making use of mean and standard deviation has been checked and it does not make any sense as the simulation
#cannot be performed using Normalization. Most of the values are patient generated data and so this cannot be simulated.

#Doing Correlation analysis of the whole dataset
corr_matrix <- cor(cervical_cancer)

ggplot(data = reshape2::melt(corr_matrix)) + 
  geom_tile(aes(x = Var1, y = Var2, fill = value)) + 
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal() + 
  labs(
    x = 'Variables',
    y = 'Variables',
    title = 'CORRELATION MATRIX',
    fill = 'correlation'
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



#We can find several variables have heavy correlation with every other variable which have to be removed. The heavy correlation variables are selecgted here.
#Certain variables which showed heavy correlation are removed -> Smokes, IUD, STD, STDs:condylomatosis, STDs: Number of diagnosis

cervical_cancer = cervical_cancer[,-c(5,10,12,26,14)]

#Some variables dont have proper distribution and so removing those variables as well 
#(STDs:cervical condylomatosis, STDs:vaginal condylomatosis, STDs:pelvic inflammatory disease, STDs:HPV, `STDs:Hepatitis B`, `STDs:AIDS`, `STDs:genital herpes`, `STDs:molluscum contagiosum`, )

cervical_cancer = cervical_cancer[,-c(11,12,15,16,17,18,20,21)]

palette_ro <- c("yellow", "#FC4E07")

### Age distribution

p1 <- ggplot(cervical_cancer, aes(x = Age)) + 
  geom_histogram(aes(y = ..density..), binwidth = 5, fill = palette_ro[1], color = "black") +
  geom_density(adjust = 0.8, fill = "cyan", color = "black", alpha = 0.25) + 
  scale_x_continuous(breaks = seq(10, 90, 10)) +
  geom_vline(xintercept = median(cervical_cancer$Age), linetype = "longdash", colour = "blue") +
  labs(
    x = "Age",
    y = "density",
    title = "Age Distribution"
  )
p1


# Density plot for Age with respect to Cancer

p2 = ggplot(cervical_cancer, aes(x = Age, fill = factor(`Dx:Cancer`))) +
  geom_density(alpha = 0.64) +
  scale_fill_manual(
    values = c(palette_ro[2], palette_ro[7]),
    name = "Cancer",
    labels = c("0 (No cancer)", "1 (Cancer)")
  ) +
  scale_x_continuous(breaks = seq(10, 90, 10))

grid.arrange(p1, p2, nrow = 2)

### Number of Sexual Partners distribution

p2 = ggplot(cervical_cancer, aes(x = `Number of sexual partners`, fill = factor(`Dx:Cancer`))) +
  geom_density(alpha = 0.64) +
  scale_fill_manual(
    values = c(palette_ro[2], palette_ro[7]),
    name = "Cancer",
    labels = c("0 (No cancer)", "1 (Cancer)")
  ) +
  scale_x_continuous(breaks = seq(0, 35, 5))


# Print the plot
print(p2)

### Distribution of First Sexual Intercourse

p1 <- ggplot(cervical_cancer, aes(x = `First sexual intercourse`)) + 
  geom_histogram(aes(y = ..density..), binwidth = 4, fill = palette_ro[1], color = "black") +
  geom_density(adjust = 1.0, fill = "cyan", color = "black", alpha = 0.25) + 
  scale_x_continuous(breaks = seq(5, 40, 5)) +
  geom_vline(xintercept = median(cervical_cancer$`First sexual intercourse`), linetype = "longdash", colour = "blue") +
  labs(
    x = "First sexual intercourse of patient",
    y = "density",
    title = "First Sexual Intercourse Distribution "
  )
p1


# Density plot for First sexual intercourse with respect to Cancer

p2 = ggplot(cervical_cancer, aes(x = `First sexual intercourse`, fill = factor(`Dx:Cancer`))) +
  geom_density(alpha = 0.64) +
  scale_fill_manual(
    values = c(palette_ro[2], palette_ro[7]),
    name = "Cancer",
    labels = c("0 (No cancer)", "1 (Cancer)")
  ) +
  scale_x_continuous(breaks = seq(5, 40, 5))

grid.arrange(p1, p2, nrow = 2)


### Number of pregnancies

p1 <- ggplot(cervical_cancer, aes(x = `Num of pregnancies`)) + 
  geom_histogram(aes(y = ..density..), binwidth = 1, fill = palette_ro[1], color = "black") +
  geom_density(adjust = 1.0, fill = "cyan", color = "black", alpha = 0.25) + 
  scale_x_continuous(breaks = seq(0, 12, 1)) +
  geom_vline(xintercept = median(cervical_cancer$`Num of pregnancies`), linetype = "longdash", colour = "blue") +
  labs(
    x = "Number of Pregnancies of patient",
    y = "density",
    title = "Number of Pregancies Distribution"
  )
p1

p2 = ggplot(cervical_cancer, aes(x = `Num of pregnancies`, fill = factor(`Dx:Cancer`))) +
  geom_density(alpha = 0.64) +
  scale_fill_manual(
    values = c(palette_ro[2], palette_ro[7]),
    name = "Cancer",
    labels = c("0 (No cancer)", "1 (Cancer)")
  ) +
  scale_x_continuous(breaks = seq(1, 12, 1))

grid.arrange(p1, p2, nrow = 2)


### Number of smokes consumed

p1 <- ggplot(cervical_cancer, aes(x = `Smokes (years)`)) + 
  geom_histogram(aes(y = ..density..), binwidth = 3, fill = palette_ro[1], color = "black") +
  geom_density(adjust = 1.0, fill = "cyan", color = "black", alpha = 0.25) + 
  scale_x_continuous(breaks = seq(0, 40, 5)) +
  geom_vline(xintercept = median(cervical_cancer$`Smokes (years)`), linetype = "longdash", colour = "blue") +
  labs(
    x = "Number of Smokes of patient",
    y = "density",
    title = "Number of Smokes Distribution"
  )
p1

p2 = ggplot(cervical_cancer, aes(x = `Smokes (years)`, fill = factor(`Dx:Cancer`))) +
  geom_density(alpha = 0.64) +
  scale_fill_manual(
    values = c(palette_ro[2], palette_ro[7]),
    name = "Cancer",
    labels = c("0 (No cancer)", "1 (Cancer)")
  ) +
  scale_x_continuous(breaks = seq(0, 40, 5))

grid.arrange(p1, p2, nrow = 2)

#We don't have a proper distribution of data here..

### Hormonal Contraceptives (years)

p1 <- ggplot(cervical_cancer, aes(x = `Hormonal Contraceptives (years)`)) + 
  geom_histogram(aes(y = ..density..), binwidth = 2, fill = palette_ro[1], color = "black") +
  geom_density(adjust = 1.0, fill = "cyan", color = "black", alpha = 0.25) + 
  scale_x_continuous(breaks = seq(0, 25, 5)) +
  geom_vline(xintercept = median(cervical_cancer$`Hormonal Contraceptives (years)`), linetype = "longdash", colour = "blue") +
  labs(
    x = "Hormonal Contraceptives (years)",
    y = "density",
    title = "Hormonal Contraceptives (years) Distribution"
  )
p1

p2 = ggplot(cervical_cancer, aes(x = `Hormonal Contraceptives (years)`, fill = factor(`Dx:Cancer`))) +
  geom_density(alpha = 0.64) +
  scale_fill_manual(
    values = c(palette_ro[2], palette_ro[7]),
    name = "Cancer",
    labels = c("0 (No cancer)", "1 (Cancer)")
  ) +
  scale_x_continuous(breaks = seq(0, 25, 5))

grid.arrange(p1, p2, nrow = 2)

#We don't have a proper distribution here as well..

#Lets now check the distribution of binary variables with bar charts

#Hormonal Contraceptives Distribution
p11 = ggplot(data = cervical_cancer, aes(x=as.factor(cervical_cancer$`Hormonal Contraceptives`))) +
  geom_bar(position = 'dodge', aes(fill=as.factor(cervical_cancer$`Dx:Cancer`)), color = 'black') + labs( title = 'Hormonal Contraceptives Distribution with Cancer' ) +  
  xlab('Hormonal Contraceptives') +
  scale_fill_manual(name = "Cancer", values = c("0" = "lightblue", "1" = "lightcoral"))

p11

#Schiller Distribution
p22 = ggplot(data = cervical_cancer, aes(x=as.factor(cervical_cancer$Schiller))) +
  geom_bar(position = 'dodge', aes(fill=as.factor(cervical_cancer$`Dx:Cancer`)), color = 'black') + labs( title = 'Schiller Distribution with Cancer' ) + 
  xlab('Schiller') +
  scale_fill_manual(name = "Cancer", values = c("0" = "lightblue", "1" = "lightcoral"))
p22

#Hinselmann Distribution
p33 = ggplot(data = cervical_cancer, aes(x=as.factor(cervical_cancer$Hinselmann))) +
  geom_bar(position = 'dodge', aes(fill=as.factor(cervical_cancer$`Dx:Cancer`)), color = 'black') + labs( title = 'Hinselmann Distribution with Cancer' ) + 
  xlab('Hinselmann') +
  scale_fill_manual(name = "Cancer", values = c("0" = "lightblue", "1" = "lightcoral"))

p33

#Biopsy Distribution
p44 = ggplot(data = cervical_cancer, aes(x=as.factor(cervical_cancer$Biopsy))) +
  geom_bar(position = 'dodge', aes(fill=as.factor(cervical_cancer$`Dx:Cancer`)), color = 'black') + labs( title = 'Biopsy Distribution with Cancer' ) + 
  xlab('Biopsy') +
  scale_fill_manual(name = "Cancer", values = c("0" = "lightblue", "1" = "lightcoral"))

p44

grid.arrange(p11, p22, nrow = 2, ncol = 2)

#Checking the distribution of all non-binary variables with Box-Plots

df <- melt(cervical_cancer[, c(1, 2, 3, 4, 14)], id.var = "Dx:Cancer")

ggplot(data = df, aes(x = variable, y = value)) +
  geom_boxplot(aes(fill = factor(`Dx:Cancer`))) +
  scale_fill_manual(name = "Cancer", values = c("0" = "lightblue", "1" = "lightcoral")) +
  facet_wrap(~variable, scales = "free")


#With all these information, Lets check the skewness to confirm if there are irregular distribution

#Computing the skewness for All Numerical Variables
skewness_df = data.frame(Variable_Name = character(), skewness = numeric(), stringsAsFactors =
                           FALSE)
for(i in names(cervical_cancer)){
  if(is.numeric(cervical_cancer[[i]])){
    value = skewness(cervical_cancer[[i]], na.rm=TRUE)
    skewness_df = rbind(skewness_df, data.frame(Variable_Name = i, skewness = value,
                                                stringsAsFactors = FALSE))
  }
  
}
skewness_df
ggplot(skewness_df, aes(x = skewness, y = Variable_Name)) +
  geom_bar(stat = "identity", fill = "skyblue", color = 'black') +  # You can customize the fill color
  labs(
    title = "Skewness Chart",
    x = "Skewness",
    y = "Variables"
  )

#We can find that all the variables are skewed requiring us to scale the variables for sure before doing the modeling analysis.
#The Feature Scaling and Prediction Analysis will be done using Python. Some descriptive statistical distribution between variables will be proved using SQL.

cervical_cancer$`Hormonal Contraceptives` = as.factor(cervical_cancer$`Hormonal Contraceptives`)

cervical_cancer$`STDs:vulvo-perineal condylomatosis` = as.factor(cervical_cancer$`STDs:vulvo-perineal condylomatosis`)

cervical_cancer$`STDs:syphilis` = as.factor(cervical_cancer$`STDs:syphilis`)

cervical_cancer$`STDs:HIV` = as.factor(cervical_cancer$`STDs:HIV`)

cervical_cancer$`Dx:Cancer` = as.factor(cervical_cancer$`Dx:Cancer`)

cervical_cancer$`Dx:CIN` = as.factor(cervical_cancer$`Dx:CIN`)

cervical_cancer$`Dx:HPV` = as.factor(cervical_cancer$`Dx:HPV`)

cervical_cancer$Dx = as.factor(cervical_cancer$Dx)

cervical_cancer$Hinselmann = as.factor(cervical_cancer$Hinselmann)

cervical_cancer$Schiller = as.factor(cervical_cancer$Schiller)

cervical_cancer$Citology = as.factor(cervical_cancer$Citology)

cervical_cancer$Biopsy = as.factor(cervical_cancer$Biopsy)

str(cervical_cancer)

numeric_cols <- cervical_cancer %>% select_if(is.numeric)
skewness_values <- sapply(numeric_cols, skewness)
skewness_df <- data.frame(variable = names(skewness_values), skewness = skewness_values)
skewness_df$index <- row.names(skewness_df)
rownames(skewness_df) <- NULL  # Remove row names
skewness_df1 <- skewness_df[, c( "variable", "skewness")]  # Rearrange columns
skewness_df1

ggplot(skewness_df1, aes(x = skewness, y = variable)) +
  geom_bar(stat = "identity", fill = "skyblue", color = 'black') +  # You can customize the fill color
  labs(
    title = "Skewness Chart",
    x = "Skewness",
    y = "Variables"
  )

#The variables are having very high skewness and so it is recommended to scale the variables.
#The prediction modeling process will be done in python.

#Checking the Significance of variables - Will remove variables if there is any significance

#Logistic Regression - To Check Significance
#Non-Scaled Data Model Fit
glm.model = glm(`Dx:Cancer`~.,data=cervical_cancer,family="binomial")
summary(glm.model)

#We don't have any significance. Let's check the same with all models being implemented in python.

#Save the dataset back to the system
write.csv(cervical_cancer, file = "/Users/jasonrayen/Downloads/Jason Masters/Sem 2/Applications of Metadata in Complex Big Data Problems (AIT 582)/Project/dataset_from_r.csv", row.names = FALSE)

























