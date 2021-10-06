# Plantilla para pre-procesado de datos
# Importar Datasets

datasets <- read.csv("Data.csv")
# datasets <- datasets[,2:3]

# Tratamiento de los valores NA
datasets$Age <- ifelse(is.na(datasets$Age),ave(datasets$Age, FUN = function(x) mean(x,na.rm=TRUE)),datasets$Age)
datasets$Salary <- ifelse(is.na(datasets$Salary),ave(datasets$Salary, FUN = function(x) mean(x,na.rm=TRUE)),datasets$Salary)

# Codificar variables categoricas
datasets$Country <- factor(datasets$Country, levels=c("France","Spain","Germany"),
                           labels = c(1,2,3))
datasets$Purchased <- factor(datasets$Purchased, levels=c("No","Yes"),
                           labels = c(0,1))

# dividir los conjuntos de datos de entrenamiento y testing
library("caTools")
set.seed(123)
split <- sample.split(datasets$Purchased,SplitRatio = 0.8)
training_set <- subset(datasets, split==TRUE)
testing_set <- subset(datasets, split==FALSE)

# Escalado de valores
training_set[,2:3] = scale(training_set[,2:3])
testing_set[,2:3] = scale(testing_set[,2:3])