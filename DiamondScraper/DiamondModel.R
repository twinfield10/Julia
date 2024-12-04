# DiamondModelR

# Environment
library(tidyverse)
library(ggplot2)
library(caret)
library(xgboost)
library(fastDummies)
library(randomForest)
set.seed(42)


# Load Raw Data
raw <- read.csv("./data/2024-12-03 05-57 PM.csv") %>% filter(carat >=1) %>% glimpse()

# Categorical Values
all_shapes <- c('Round', 'Oval', 'Emerald', 'Cushion', 'Elongated Cushion',
                'Pear', 'Radiant', 'Princess', 'Marquise', 'Asscher', 'Heart')
all_cuts <- c('Super Ideal', "Ideal", "Very Good", "Good", "Fair")
all_color <- c('D', 'E', 'F', 'G', 'H', 'I', 'J')
all_clarity <- c('FL', 'IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2')

# Preprocess
prep_data <- function(df){
  
  # Check NA Rows
  na_rows <- sum(is.na(df))
  if(na_rows > 0){
    print(paste0(na_rows, " Rows With NA Values"))
    df %>% filter(is.na(any())) %>% print.data.frame()
  }
  # Check Dupe Rows
  dupes <- df %>% filter(duplicated(.))
  if(nrow(dupes) > 0){
    print(paste0(nrow(dupes) ," Duplicated Rows"))
    dupes %>% print.data.frame()
  }
  
  id_cols <- c('id', 'upc', 'url')
  cat_cols <- c('shape', 'cut', 'color', 'clarity', 'report', 'origin', 'polish', 'symmetry', 'fluorescence', 'has_cert')
  cont_cols <- c('carat', 'length_width_ratio', 'depth', 'table')
  
  # Handle Categorical Variables
  df <- df %>%
    mutate(across(all_of(cat_cols), as.factor)) %>%
    fastDummies::dummy_cols(select_columns = cat_cols, remove_first_dummy = FALSE)
  
  # Remove ID and URL Columns
  df <- df %>%
    select(-cat_cols)
  
  # Clean Column Names
  colnames(df) <- gsub(" ", "_", colnames(df))
  
  return(df)
}
data <- prep_data(raw)
data %>% glimpse()


# Plot By carat
ggplot(raw %>% filter(carat >= 1.5 & carat <= 3.5),
       aes(x = carat, y = price, color=shape)) +
  geom_point() +
  theme_minimal() +
  labs(
    title = "Loose Diamond Price (y) by Carat (x)",
    subtitle = "Data From Brilliant Earth"
  )

# Build Model - Split Data
ignore_cols <- c('id', 'upc', 'url', 'measurements', 'date_fetched', 'type')
train_indices <- createDataPartition(data$price, p = 0.8, list = FALSE)
train_data <- data[train_indices, ] %>% select(-all_of(ignore_cols))
test_data <- data[-train_indices, ]

## 1)  Linear
model_lm <- lm(price ~ ., data = train_data)
summary(model_lm)
# R-Squared = 0.8398


## 2) Random Forest
model_rf <- randomForest(price ~ ., data = train_data, importance = TRUE)


## 3) xGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_data %>% select(-price)), label = train_data$price)
params <- list(objective = "reg:squarederror", eta = 0.1, max_depth = 6)
model_xgb <- xgboost(params = params, data = dtrain, nrounds = 100)

## 4) Boosted Linear Model
linear_xgb_build <- function(){
  raw_train_indices <- createDataPartition(raw$price, p = 0.8, list = FALSE)
  raw_train_data <- raw[raw_train_indices, ]
  raw_test_data <- raw[-raw_train_indices, ]
  
  train_data_processed <- raw_train_data %>%
    select(-url, -date_fetched) %>%
    mutate_if(is.character, as.factor) %>%
    mutate_if(is.factor, as.numeric)
  
  test_data_processed <- raw_test_data %>%
    select(-url, -date_fetched) %>%
    mutate_if(is.character, as.factor) %>%
    mutate_if(is.factor, as.numeric)
  
  test_ids <- test_data_processed$id
  
  train_matrix <- as.matrix(train_data_processed %>% select(-price, -id))
  train_label <- train_data_processed$price
  test_matrix <- as.matrix(test_data_processed %>% select(-price, -id))
  test_label <- test_data_processed$price
  
  # Set parameters for XGBoost with linear booster
  params <- list(
    booster = "gbtree",  # Use linear booster
    objective = "reg:squarederror",  # Regression task (predicting price)
    eta = 0.1,  # Learning rate (default is 0.3)
    nthread = 4  # Number of threads (you can adjust this)
  )
  
  # Train the XGBoost model
  lm_model_xgb <- xgboost(
    data = train_matrix, 
    label = train_label, 
    params = params, 
    nrounds = 400,  # Number of boosting iterations
    verbose = 0  # Show progress during training
  ) 
  
  # Make predictions on the test data
  predictions <- predict(lm_model_xgb, newdata = test_matrix)
  
  # Evaluate the model (e.g., using RMSE)
  rmse <- sqrt(mean((predictions - test_label)^2))
  print(paste("RMSE: ", rmse))
  
  # Compare actual vs predicted prices
  comparison <- data.frame(
    id = test_ids,
    Actual = test_label,
    Predicted = predictions,
    Difference = abs(test_label - predictions)
  )
  print(head(comparison))
  
  # Plot Importance
  importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = lm_model_xgb)
  xgb.plot.importance(importance_matrix = importance_matrix, top_n = 10)
}
linear_xgb_build()


### Predict and Analyze
lm_predictions <- predict(model_lm, newdata = test_data)
rf_predictions <- predict(model_rf, newdata = test_data)
xgb_predictions <- predict(model_xgb, newdata = as.matrix(test_data %>% select(-price) %>% select(-all_of(ignore_cols))))

library(Metrics)
eval_models <- function(){
  # Actual Prices
  actual <- test_data$price
  
  # Linear Model Metrics
  lm_mae <- mae(actual, lm_predictions)
  lm_mse <- mse(actual, lm_predictions)
  lm_rmse <- rmse(actual, lm_predictions)
  lm_r2 <- cor(actual, lm_predictions)^2
  
  # Random Forest Metrics
  rf_mae <- mae(actual, rf_predictions)
  rf_mse <- mse(actual, rf_predictions)
  rf_rmse <- rmse(actual, rf_predictions)
  rf_r2 <- cor(actual, rf_predictions)^2
  
  # XGBoost Metrics
  xgb_mae <- mae(actual, xgb_predictions)
  xgb_mse <- mse(actual, xgb_predictions)
  xgb_rmse <- rmse(actual, xgb_predictions)
  xgb_r2 <- cor(actual, xgb_predictions)^2
  
  # Print out the metrics for comparison
  cat("Linear Model:\n")
  cat("  MAE:", lm_mae, "\n  MSE:", lm_mse, "\n  RMSE:", lm_rmse, "\n  R²:", lm_r2, "\n\n")
  
  cat("Random Forest Model:\n")
  cat("  MAE:", rf_mae, "\n  MSE:", rf_mse, "\n  RMSE:", rf_rmse, "\n  R²:", rf_r2, "\n\n")
  
  cat("XGBoost Model:\n")
  cat("  MAE:", xgb_mae, "\n  MSE:", xgb_mse, "\n  RMSE:", xgb_rmse, "\n  R²:", xgb_r2, "\n")
}
eval_models()

plot_evals <- function(){
  # Actual Prices
  actual <- test_data$price
  
  # Combine predictions
  comparison_df <- data.frame(
    actual = actual,
    lm_predictions = lm_predictions,
    rf_predictions = rf_predictions,
    xgb_predictions = xgb_predictions
  )
  
  # Plot Actual vs Predicted for each model
  ggplot(comparison_df) +
    geom_point(aes(x = actual, y = lm_predictions), color = "blue", alpha = 0.5) +
    geom_point(aes(x = actual, y = rf_predictions), color = "red", alpha = 0.5) +
    geom_point(aes(x = actual, y = xgb_predictions), color = "green", alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = "Actual vs Predicted: Linear, Random Forest, XGBoost",
         x = "Actual Price", y = "Predicted Price") +
    scale_color_manual(name = "Model", values = c("blue", "red", "green"))
}
plot_evals()

# Build Inputs
best_deals <- function(
    Shape = all_shapes, #c("Cushion", 'Elongated Cushion', 'Oval'),
    Carat_min = 0,
    Carat_max = 50,
    Color = all_color,
    Cut = NULL,
    Clarity = NULL,
    Type = NULL,
    LW_Ratio_min = 1,
    Price_max = 10000000 
) {
  # Filter raw data to match the input conditions
  filtered_data <- raw %>%
    filter(shape %in% Shape
           ,color %in% Color
           ,carat <= Carat_max
           ,carat >= Carat_min
           ,length_width_ratio >= LW_Ratio_min
           ,price <= Price_max
           )
  
  # Check if filtering returned any rows
  if (nrow(filtered_data) == 0) {
    stop("No matching diamonds found in the raw data for the specified conditions.")
  }
  
  # Prep
  clean_filt <- prep_data(filtered_data)
  missing_cols <- setdiff(names(test_data), names(clean_filt))
  
  for (col in missing_cols) {
    clean_filt[[col]] <- 0
  }
  clean_filt <- clean_filt %>% select(names(test_data))
  
  
  # Make Predictions
  xlm_predictions <- predict(model_lm, newdata = clean_filt)
  xrf_predictions <- predict(model_rf, newdata = clean_filt)
  xxgb_preds <- predict(model_xgb, newdata = as.matrix(clean_filt %>% select(-price) %>% select(-all_of(ignore_cols))))
  
  # Clean Final
  Clean <- filtered_data
  Clean$xgb_pred <- xxgb_preds
  Clean$rf_pred <- xrf_predictions
  Clean$lm_pred <- xlm_predictions
  
  # Differences
  Clean$xgb_resid <- Clean$xgb_pred - Clean$price
  Clean$rf_resid <- Clean$rf_pred - Clean$price
  Clean$lm_resid <- Clean$lm_pred - Clean$price
  
  # Clean
  Clean <- Clean %>%
    select(id, report, shape, carat, color, clarity, cut, type, length_width_ratio, price, rf_pred, rf_resid) %>%
    arrange(-rf_resid) %>%
    mutate(
      xDiscount =  scales::percent(rf_resid / rf_pred, accuracy = 0.01),
      across(c(price, rf_pred, rf_resid), ~ scales::dollar(.))
    ) %>%
    rename(
      Pred_Price = rf_pred,
      Pred_Diff = rf_resid
    )
  
  cat("Best Deals: \n")
  print(Clean %>% head(10))
  
  #cat("Best Deals on Ovals: \n")
  #print(Clean %>% filter(shape == 'Oval') %>% head(10))
  #cat("\n")
  #cat("Best Deals on Cushions: \n")
  #print(Clean %>% filter(str_detect(shape, "Cushion")) %>% head(10))
  #cat("\n")
  
}
best_deals(
  Shape = c('Oval', 'Cushion'),
  Carat_min = 2.25,
  Carat_max = 3.5,
  Color = c('D', 'E', 'F', 'G', 'H', 'I'),
  #Clarity = all_clarity,
  LW_Ratio_min = 1.3,
  Price_max = 20000
)


# Test Inputs
create_input_tbl <- function(
    Shape = "Round",
    Carat_min = 2.75,
    Carat_max = 3.5,
    Color = "H",
    Cut = NULL,
    Clarity = NULL,
    Type = NULL
) {
  # Filter raw data to match the input conditions
  filtered_data <- raw %>%
    filter(shape == Shape & color == Color & carat <= Carat_max & carat >= Carat_min)
  
  # Check if filtering returned any rows
  if (nrow(filtered_data) == 0) {
    stop("No matching diamonds found in the raw data for the specified conditions.")
  }
  
  clean_filt = prep_data(filtered_data)
  clean_filt %>% glimpse()
  
  clean_sum <- clean_filt %>%
    summarise(across())

  
  # Build Raw Input Table
  input <- data.frame(
    id = c(99999999),
    upc = c("TEST"),
    url = c("TEST"),
    shape = c(Shape),
    price = c(10000),
    carat = c(Carat),
    cut = c(Cut),
    color = c(Color),
    clarity = c(Clarity),
    report = c('GIA'),
    origin = c('Botswana Sort'),
    polish = c('Excellent'),
    symmetry = c('Excellent'),
    measurements = c('TEST'),
    fluorescence = c('None'),
    has_cert = c('True'),
    length = c(6.390),
    length_width_ratio = c(1.4),
    depth = c(60),
    table = c(60),
    type = c(Type),
    date_fetched = c("TEST")
  )
  

  df <- rbind(raw, input)
  
  model_df <- prep_data(df) %>%
    filter(id == 99999999) %>%
    select(all_of(names(train_data))) %>%
    select(-price)
  
  # Apply Model
  input_matrix <- as.matrix(model_df)
  
  # BootStrap Itr
  n_iterations <- 1000
  bootstrap_preds <- numeric(n_iterations)
  for (i in 1:n_iterations) {
    # Resample with replacement
    resample <- sample(nrow(input_matrix), replace = TRUE)
    input_resampled <- input_matrix[resample, , drop = FALSE]
    bootstrap_preds[i] <- predict(model_xgb, newdata = input_resampled)
  }
  lower_bound <- quantile(bootstrap_preds, 0.025)
  upper_bound <- quantile(bootstrap_preds, 0.975)
  
  predicted_price <- predict(model_xgb, newdata = input_matrix)
  
  
  # Print Statements
  cat("Diamond Specs:\n")
  cat("Shape:", input$shape, "\n")
  cat("Carat:", input$carat, "\n")
  cat("Cut:", input$cut, "\n")
  cat("Color:", input$color, "\n")
  cat("Clarity:", input$clarity, "\n")
  cat("Type:", input$type, "\n")
  cat("L/W Ratio:", input$length_width_ratio)
  cat("\n")
  cat("Projected Price:", scales::dollar(predicted_price), "\n")
  cat("Confidence Interval: [", scales::dollar(lower_bound), ", ", scales::dollar(upper_bound), "]\n")
}

create_input_tbl()



