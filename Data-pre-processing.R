require(caret)
require(dplyr)

# 指数表示の回避
options(scipen=10)

#
# データの読み込み
#
train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")

# train/test のタグ付け
train$data <- "train"
test$data <- "test"

# train と test をマージ
all <- bind_rows(train, test)

# 目的変数の factor 化
all[which(all$y == 1 & all$data == "train"),"y"] <- "yes"
all[which(all$y == 0 & all$data == "train"),"y"] <- "no"
all[which(all$data == "test"),"y"] <- "none"

# 目的変数名を response に変更
names(all)[names(all) == "y"] <- "response"

#
# all$age2 <- cut(all$age, breaks=c(17,30,34,39,45,52,95))
# all$duration2 <- cut(all$duration, breaks=c(-1,80,127,181,260,415,4918))
# all$balance2 <- cut(all$balance,breaks=c(-6848,2,175,449,953,2281,102127))
# all$pdays2 <- cut(all$pdays, breaks=c(-2,0,1,91,871))

#
# Dummy 変数なし
#

# all から train のデータを抽出
all.train <- all[which(all$data == "train"),]
all.train$response <- as.factor(all.train$response)

# 不要な列: data を
all.train <- subset(all.train, select = -c(data))

# 再現性のため乱数シードを固定
set.seed(10)

# 訓練データと検証データに分割する
# Train 用の列番号を作成
inTrain <- caret::createDataPartition(all.train$response, p = .8, list = FALSE)
train.train <- all.train[inTrain,]
train.test <- all.train[-inTrain,]

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
# Dummy 変数なし ( nearZeroVar() 適用 )
#

all.response <- subset(all, select = c(response))
all.Descr <- subset(all, select = -c(response))

# 情報量が少ない ( 分散がほぼ 0 ) の変数を除く
nzv <- caret::nearZeroVar(all.Descr)

names(all)[nzv]
#[1] "default" "pdays" 

filterdDescr <- all.Descr[,-nzv]
all.nzv <- cbind(filterdDescr,all.response)

# all から train のデータを抽出
all.nzv.train <- all.nzv[which(all.nzv$data == "train"),]
all.nzv.train$response <- as.factor(all.nzv.train$response)

# 不要な列: data を
all.nzv.train <- subset(all.nzv.train, select = -c(data))

# 再現性のため乱数シードを固定
set.seed(10)

# 訓練データと検証データに分割する
# Train 用の列番号を作成
inTrain <- caret::createDataPartition(all.nzv.train$response, p = .8, list = FALSE)
train.nzv.train <- all.nzv.train[inTrain,]
train.nzv.test <- all.nzv.train[-inTrain,]

dim(train.nzv.train)
# [1] 21704    16

train.nzv.train$response %>% table
# .
# no   yes 
# 19164  2540 

dim(train.nzv.test)
# [1] 5424   16

train.nzv.test$response %>% table
# .
# no  yes 
# 4790  634 

#
# Dummy 変数化
#

noNames <- caret::dummyVars(~., data = subset(all, select = -c(response)), fullRank = FALSE  )
all.dummy <- as.data.frame(predict(noNames, all))
all.dummy <- data.frame(all.dummy, response = all$response)


#
# Dummy ( 前処理なし )
#

# データの分割
train.dummy <- all.dummy[which(all.dummy$datatrain == 1),]
train.dummy <- subset(train.dummy, select = -c(datatest, datatrain))
train.dummy$response <- train.dummy$response[,drop = TRUE]

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


#
# Dummy ( nearZeroVar(), findCorrelation() 適用 )
#

all.dummy.response <- subset(all.dummy, select = c(response))
all.dummy.Descr <- subset(all.dummy, select = -c(response, datatest, datatrain))
all.dummy.data <- subset(all.dummy, select = c(datatest, datatrain))

# 情報量が少ない ( 分散がほぼ 0 ) の変数を除く
nzv <- caret::nearZeroVar(all.dummy.Descr)

names(all.dummy)[nzv]
# [1] "job.entrepreneur"  "job.housemaid"     "job.self.employed" "job.student"       "job.unemployed"    "job.unknown"      
# [7] "education.unknown" "default.no"        "default.yes"       "month.dec"         "month.jan"         "month.mar"        
# [13] "month.oct"         "month.sep"         "pdays"             "poutcome.other"    "poutcome.success" 

filterdDescr.dummy <- all.dummy.Descr[,-nzv]

# 相関が強い変数を削除
descCor.dummy <- cor(filterdDescr.dummy)
highlyCorDescr.dummy <- findCorrelation(descCor.dummy, cutoff = 0.75)
names(filterdDescr.dummy)[highlyCorDescr.dummy]
# [1] "month.feb"        "contact.cellular" "marital.single"   "day" 

filterdDescr.dummy <- filterdDescr.dummy[,-highlyCorDescr.dummy]

all.dummy.nzv.highlyCorDescr <- cbind(filterdDescr.dummy, all.dummy.response, all.dummy.data)

# データの分割
train.dummy.nzv.highlyCorDescr <- all.dummy.nzv.highlyCorDescr[which(all.dummy.nzv.highlyCorDescr$datatrain == 1),]
train.dummy.nzv.highlyCorDescr <- subset(train.dummy.nzv.highlyCorDescr, select = -c(datatest, datatrain))
train.dummy.nzv.highlyCorDescr$response <- train.dummy$response[,drop = TRUE]

test.dummy.nzv.highlyCorDescr <- all.dummy.nzv.highlyCorDescr[which(all.dummy.nzv.highlyCorDescr$datatest == 1),]
test.dummy.nzv.highlyCorDescr <- subset(test.dummy.nzv.highlyCorDescr, select = -c(datatest, datatrain, response ))

# 再現性のため乱数シードを固定
set.seed(10)

# 訓練データと検証データに分割する
# Train 用の列番号を作成
inTrain.dummy.nzv.highlyCorDescr  <- caret::createDataPartition(train.dummy.nzv.highlyCorDescr$response, p = .8, list = FALSE)
train.dummy.nzv.highlyCorDescr.train <- train.dummy.nzv.highlyCorDescr[inTrain.dummy.nzv.highlyCorDescr,]
train.dummy.nzv.highlyCorDescr.test <- train.dummy.nzv.highlyCorDescr[-inTrain.dummy.nzv.highlyCorDescr,]

dim(train.dummy.nzv.highlyCorDescr.train)
# [1] 21704    28

train.dummy.nzv.highlyCorDescr.train$response %>% table
# .
# no   yes 
# 19164  2540 

dim(train.dummy.nzv.highlyCorDescr.test)
# [1] 5424   28

train.dummy.nzv.highlyCorDescr.test$response %>% table
# .
# no  yes 
# 4790  634 


#
# オリジナルの test/train に追加した列: data を削除
#
test <- subset(test, select = -c(data))
train <- subset(train, select = -c(data))
