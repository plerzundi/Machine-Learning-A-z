# SVR

# Importar el dataset
dataset = read.csv("Position_Salaries.csv")
dataset = dataset[,2:3]

set.seed(123)

# Ajustar el modelo SVR con el conjunto de datos
#install.packages("e1071")
library(e1071)

regression = svm(formula = Salary ~ ., data=dataset,
                 type = "eps-regression",
                 kernel="radial")


# Prediccion de nuevos resultados con SVR
y_pred = predict(regression, newdata = data.frame(Level = 6.5))

# Visualizacion del modelo de DVR
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), color="red") +
  geom_line(aes(x = dataset$Level, y = predict(regression,newdata = data.frame(Level = dataset$Level))),
                color = "blue") +
  ggtitle("Prediccion (SVR)")+
  xlab("Nivel del empleado")+
  ylab("Sueldo (en $)")