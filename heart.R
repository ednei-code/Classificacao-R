#Analise de Classificação - Classificar os pacientes que desenvolvem doencas cardiacas?

# Neste projeto, vamos prever quais pacientes podem desenvolver doenças cardiacas. 
# Usaremos a regressão logística, a Ãrvore de decisÃo e a floresta aleatória como modelos de Machine Learning. 


# Meu objetivo sera analisar e compreender as razões pelas quais um paciente desenvolve doenças cardiacas 
# estudar a relação entre as muitas variaveis disponiveis.


#========================================================================================================================================

setwd("C:/portfolioR")
getwd()

#pacotes
library(plyr)
library(dplyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)
library(e1071)

#https://archive.ics.uci.edu/ml/datasets/Heart+Disease
#carregando os dados
df = read.csv("heart/heart.csv",sep=',')

dim(df)
str(df)
View(df)
df_copy <- df
View(df_copy)
#checando valores missing
sapply(df,function(x) sum(is.na(x)))
sum(is.na(df))


#as variaveis (sex,cp,slope,target) devem ser fator

df$sex <- factor(df$sex)
df$cp <- factor(df$cp)
df$slope <- factor(df$slope)
df$target <- factor(df$target)
df$fbs <- factor(df$fbs)
df$restecg <- factor(df$restecg)
df$exang <- factor(df$exang)
df$thal <- factor(df$thal)
df$ca <- factor(df$ca)
df$exang <- factor(df$exang)


colnames(df$ï..age) <- df$age
?colnames
?data.frame


#verificando a proporção da variavel target 1=doença ; 0=ñ doença
table(df$target)
prop.table(table(df$target)) * 100

#o mesmo resultado em formato de tabela dplyr
tabela_target <- df %>% group_by(target) %>%
  summarise(Total = length(target)) %>%
  mutate(Taxa = Total / sum(Total) *100.0)
print(tabela_target)


#plot da variavel target (0)significa doença e (1)significa não doenca
ggplot(tabela_target, aes(x = '', y = Taxa, fill = target)) +
  geom_bar(width = 1, size = 1, color = 'black', stat = 'identity') +
  coord_polar('y') +
  geom_text(aes(label = paste0(round(Taxa), '%')), 
            position = position_stack(vjust = 0.5)) +
  scale_fill_manual(values = c("#777777", "#E69F00"))+
  labs(title = 'Taxa de Pacientes com doença cardiaca') +
  theme_classic() +
  theme(axis.line = element_blank(),axis.title.x = element_blank(),axis.title.y = element_blank(),
        axis.ticks = element_blank(),
        axis.text = element_blank())


#preparando o grafico para o grid para comparar as variaveis com a variavel target
b1 <- ggplot(df, aes(sex, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b2 <- ggplot(df, aes(cp, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b3 <- ggplot(df, aes(fbs, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b4 <- ggplot(df, aes(restecg, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b5 <- ggplot(df, aes(exang, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b6 <- ggplot(df, aes(slope, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b7 <- ggplot(df, aes(ca, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))
b8 <- ggplot(df, aes(thal, fill = target)) + geom_bar(position='fill') + scale_fill_manual(values=c("#999999", "#E69F00"))

#criando o grid
grid.arrange(b1,b2,b3,b4 , ncol=2)
grid.arrange(b5,b6,b7,b8 ,ncol=2)

?plot
#a variavel cp=tipo de dor no peito no grafico valor=1 e valor=3 ;juntamente com variavel thalach(frequencia cardiaca)
plot(df$cp,df$thalach,col='red',xlab='Tipo dor no peito',ylab='Frequencia cardiaca')

#variavel cp(tipo dor no peito) com a variavel trestbps(pressao arterial)
plot(df$cp,df$trestbps,col='red',xlab='Tipo dor no peito',ylab='Pressao arterial')


#correlação
cor_data = data.frame(df_copy)
corr <- cor(cor_data)

str(cor_data)
cor_data$df.cp <- as.integer(cor_data$df.cp)

corr <- cor(cor_data)
corrplot(corr,method = 'number')

#Analisando a variavel cp(tipo de dor no peito)
ggplot(data = df_copy , mapping = aes(x = cp))+
  geom_histogram(binwidth = 0.5) +
  labs(x = 'Tipo de dor Peito',
       title = 'Histograma Doença Cardiaca (tipo de dor no peito)')



 #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#modelagem preditiva
intriam <- createDataPartition(df$target,p=0.7,list = FALSE)

train <- df[intriam,]
test <- df[-intriam,]
#verificar se a divisão esta correta
dim(train) ; dim(test)

# Treinando o modelo de regressÃo logistica
# Fitting do Modelo

logmodel <- glm(target~.,family = binomial(link = 'logit'),data=train)
print(summary(logmodel))

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#analise de variancia ANOVA
anova(logmodel,test = "Chisq")
#No teste de ANOVA(teste de significancia cp,thalach,ca)são as mais significativas



test$target <- as.character(test$target)
fitted.results <- predict(logmodel, newdata = test,type = 'response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misclassificerror <- mean(fitted.results != test$target)
print(paste('Logistic Regression Accuracy',1-misclassificerror))

#Confusion Matrix da Regressao logistica
print("Confusion Matrix da Regressao Ligistica");table(test$target,fitted.results > 0.5)

# Odds Ratio

# Uma das medidas de desempenho interessantes na regressÃo logistica é Odds Ratio. 
# Basicamente, odds ratio é a chance de um evento acontecer.
exp(cbind(OR=coef(logmodel),confint(logmodel)))


#===================================================================================================================================
#arvores de decisão
#vou usar somente 4 variaveis('cp','ca','thalach','trestbps')
tree <- ctree(target ~ cp+ca+thalach+trestbps,train)
plot(tree, type = 'simple')


pred_tree <- predict(tree, test)
print('Confusion Matrix para Decision Tree'); table(Pred = pred_tree, Actual = test$target)


p1 <- predict(tree , train)
tab1 <- table(Pred = p1,Actual=train$target)
tab2 <- table(Pred = pred_tree,Actual = test$target)
print(paste('Decision Tree Accuracy', sum(diag(tab1))/sum(tab2)))



#====================================================================================================================================
#Random Forest
rf_model <- randomForest(target  ~.,data=train)
print(rf_model)
plot(rf_model)



pred_rf <- predict(rf_model , test)


print('Confusion Matrix para Randon Forest');table(test$target,pred_rf)

?varImpPlot
varImpPlot(rf_model,sort = T,n.var = 10,main = 'Top 10 Variable importance')

















































