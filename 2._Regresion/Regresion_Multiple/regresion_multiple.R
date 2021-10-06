# lectura datasets
datasets <- read.csv("50_Startups.csv")

# Codificar las variables categoricas
datasets$State = factor(datasets$State,
                        levels = c("New York","California","Florida"),
                        labels = c(1,2,3))

#Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools)
set.seed(123)
split <- sample.split(datasets$Profit,SplitRati=0.8)
training_set <- subset(datasets,split==TRUE)
testing_set <- subset(datasets,split==FALSE)

# Ajustar el modelo de Regresión Lineal Múltiple con el conjunto de entrenamiento
regression <- lm(formula = Profit ~ ., data = training_set)

# Predecir los resultados con el conjunto de testing
y_pred <- predict(regression, newdata = testing_set)

# Contruir un modelo optimo hacia atrás
SL = 0.05
regression <- lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend+State, data = datasets)
summary(regression)

regression <- lm(formula = Profit ~ R.D.Spend+Administration+Marketing.Spend, data = datasets)
summary(regression)

regression <- lm(formula = Profit ~ R.D.Spend+Administration, data = datasets)
summary(regression)

regression <- lm(formula = Profit ~ R.D.Spend, data = datasets)
summary(regression)


# FUNCION AUTOMATICA P VALOR
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)

