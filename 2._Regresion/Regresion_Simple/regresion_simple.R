# Plantilla para pre-procesado de datos
# Importar Datasets

datasets <- read.csv("Salary_Data.csv")

# Dividir los datos en conjunto de entrenamiento y conjunto de test
library(caTools)
set.seed(123)
split <- sample.split(datasets$Salary, SplitRatio = 0.8)
training_set <- subset(datasets, split == TRUE)
testing_set <- subset(datasets, split == FALSE)


# Ajustar el modelo de regresión lineal simple con el conjunto de entrenamiento
regressor <- lm(formula =Salary ~ YearsExperience, data = training_set)

# Predecir el resultado con el conjunto de test
y_pred <- predict(regressor, newdata = testing_set)

# Visualización de los resultados en el conjunto de entrenamiento
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience,training_set$Salary),
             colour="red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour ="blue") +
  ggtitle("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")+
  xlab("Año de experiencia") +
  ylab("Sueldo ( en $)")

# Visualización de los resultados en el conjunto de testing
ggplot() +
  geom_point(aes(x = testing_set$YearsExperience,testing_set$Salary),
             colour="red") +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour ="blue") +
  ggtitle("Sueldo vs Años de experiencia (Conjunto de testing)")+
  xlab("Año de experiencia") +
  ylab("Sueldo ( en $)")