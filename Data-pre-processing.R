require(caret)
require(dplyr)

# 指数表示の回避
options(scipen=10)

#
# データの読み込み
#
train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")

#
# Dummy 変数なし
#

# 目的変数の factor 化
train[which(train$y == 1),"y"] <- "yes"
train[which(train$y == 0),"y"] <- "no"
train$y <- as.factor(train$y)

# 目的変数名を response に変更
names(train)[names(train) == "y"] <- "response"

# 再現性のため乱数シードを固定
set.seed(10)

# 訓練データと検証データに分割する
# Train 用の列番号を作成
inTrain <- caret::createDataPartition(train$response, p = .8, list = FALSE)
train.train <- train[inTrain,]
train.test <- train[-inTrain,]

dim(train.train)
# [1] 21704    18

train.train$response %>% table
# .
# no   yes 
# 19164  2540 

dim(train.test)
# [1] 5424   18

train.test$response %>% table
# .
# no  yes 
# 4790  634 

#
# Dummy 変数あり
#

train$data <- "train"
test$data <- "test"

# train 
all <- bind_rows(train, test)


# dummy 変数化
noNames <- caret::dummyVars(~., data = subset(all, select = -c(response)), fullRank = FALSE  )
all.dummy <- as.data.frame(predict(noNames, all))
all.dummy <- data.frame(all.dummy, response = all$response)

# 
train.dummy <- all.dummy[which(all.dummy$datatrain == 1),]
train.dummy <- subset(train.dummy, select = -c(datatest, datatrain))

test.dummy <- all.dummy[which(all.dummy$datatest == 1),]
test.dummy <- subset(test.dummy, select = -c(datatest, datatrain, response ))


# 再現性のため乱数シードを固定
set.seed(10)

# 訓練データと検証データに分割する
# Train 用の列番号を作成
inTrain.dummy <- caret::createDataPartition(train.dummy$response, p = .8, list = FALSE)
train.dummy.train <- train.dummy[inTrain.dummy,]
train.dummy.test <- train.dummy[-inTrain.dummy,]

dim(train.dummy.train)
# [1] 21704    53

train.dummy.train$response %>% table
# .
# no   yes 
# 19164  2540 

dim(train.dummy.test)
# [1] 5424   53

train.dummy.test$response %>% table
# .
# no  yes 
# 4790  634 

test <- subset(test, select = -c(data))
train <- subset(train, select = -c(data))
