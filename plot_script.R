install.packages("tidyverse")
install.packages("ggplot2")

library(tidyverse)
library(ggplot2)


############################################################################################################
########################################### PREDICTIONS !!! ##############################################
###########################################################################################################
# Load data
predictions <- read.csv('/work/exam_repo/PredictingAbsence_DataScience2024/RegMod_Performance/predictions_test_latest.csv',sep=';')

# Calculate the difference between 'Predicted Values' and 'True Values'
predictions <- predictions %>%
  mutate(Model = if_else(Model == "dummy_mean", "Baseline Mean", 
                         if_else(Model == "dummy_yesterday", "Baseline Yesterday", 
                                 if_else(Model== "lin_reg", "Linear Regressor", 
                                         ifelse(Model == "Ridge", "Ridge Regressor", Model)))), 
         Difference = `Predicted.Values` - `True.Values`,
         Index = row_number()) %>%
  group_by(Model) %>%
  mutate(model_index = row_number()) %>%
  ungroup()

# Filter to keep only the first 100 entries for each model based on 'model_index'
filtered_predictions <- predictions %>%
  group_by(Model) %>%
  filter(model_index %in% c(0:100)) %>%
  ungroup()

# Plotting the difference over the model index
p <- ggplot(filtered_predictions, aes(x = model_index, y = Difference, color = Model, group = Model)) +
        geom_line(size = 0.2) +
        geom_point(alpha = 0.3, size = 0.2) +
        labs(title = 'A) Difference Between Predicted and True Values across Models',
            x = 'Index',
            y = 'Difference') +
        theme_minimal() +
        scale_color_manual(values = c('blue', 'navy', '#40B94D', "darkgreen", '#ADD8E6', 'purple', 'black', '#20A7AB', 'orange')) +
        theme(legend.title = element_blank(),
              text = element_text(family = "Times New Roman"),
              plot.title = element_text(size = 18, face = "bold")) +
        guides(color = guide_legend(override.aes = list(size = 3)))
p

ggsave("/work/exam_repo/PredictingAbsence_DataScience2024/plots/dif_pred_plot.jpg", plot = p, width = 10, height = 6, units = "in", dpi = 300)


#### true values plot
true_val <- read.csv('/work/exam_repo/PredictingAbsence_DataScience2024/plots/true100val.csv',sep=';')
true_val <- true_val %>% 
  mutate(model_index = row_number()) 

p_true <- ggplot(true_val, aes(x = model_index, y = True.Values)) +
  geom_line(size = 0.2, color = '#097D80') +
  geom_point(alpha = 0.3, size = 0.2) +
  scale_color_manual(values = '#097D80') +
  labs(title = 'B) True Values',
       x = 'Index',
       y = 'Difference') +
  theme_minimal() +
  theme(legend.title = element_blank(),
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(size = 18, face = "bold")) +
  guides(color = guide_legend(override.aes = list(size = 3)))
p_true

ggsave("/work/exam_repo/PredictingAbsence_DataScience2024/plots/plot_true.jpg", plot = p_true, width = 10, height = 6, units = "in", dpi = 300)



#################################################################################
########################### RESIDUAL PLOTS #####################################
#################################################################################
err <- predictions %>%
  ggplot()+
  geom_density(aes(x=Difference)) +
  labs(title = 'Distribution of Errors',
       x = 'Errors',
       y = 'Density') +
  theme_minimal() +
  xlim(-25,25)+
  scale_color_manual(values = '#097D80') +
  theme(legend.title = element_blank(),
        text = element_text(size = 13, family = "Times New Roman"),
        plot.title = element_text(size = 18, face = "bold")) +
  facet_wrap(~Model)+
  guides(color = guide_legend(override.aes = list(size = 3)))
err

ggsave("/work/exam_repo/PredictingAbsence_DataScience2024/plots/error_dist.jpg", plot = err, width = 10, height = 6, units = "in", dpi = 300)



#################################################################################
########################### FEATURE IMPORTANCE ##################################
#################################################################################
feature_imp <- read_csv('/work/DataScienceExam2024/Data/feature_importances_RandomForestRegressor.csv')



# Plot using ggplot2
p2 <- ggplot(feature_imp, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col(fill = "#097D80") +
  geom_errorbar(aes(ymin = Importance - Std, ymax = Importance + Std), width = 0.2) +
  labs(title = "Permutation Feature Importances with Random Forest",
       x = NULL,  
       y = "Mean accuracy decrease") +
  coord_flip() +  
  theme_minimal() +
  theme(legend.title = element_blank(),
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(size = 18, face = "bold")) 

# Print the plot
p2

########################### COMBINED PLOT ##################################
feature_imp_XGb <- read_csv('/work/DataScienceExam2024/Data/feature_importances_XGBRegressor.csv')

# Add a model column to each dataframe
feature_imp$Model <- "Random Forest"
feature_imp_XGb$Model <- "XGBoost"

# Combine the dataframes
combined_feature_imp <- rbind(feature_imp, feature_imp_XGb)


# Plot using ggplot2 with color distinction for each model
p_combined <- ggplot(combined_feature_imp, aes(x = reorder(Feature, Importance), y = Importance, fill = Model)) +
  geom_col(position = position_dodge(width = 0.9)) +  
  geom_errorbar(aes(ymin = Importance - Std, ymax = Importance + Std), 
                position = position_dodge(width = 0.9), 
                width = 0.2) +
  labs(title = "Permutation Feature Importances",
       subtitle = "by Best Performing Models",
       x = NULL,  
       y = "Mean Accuracy Decrease") +
  scale_fill_manual(values = c("Random Forest" = "#097D80", "XGBoost" = "#ADD8E6")) +
  coord_flip() +  
  theme_minimal() +
  theme(legend.title = element_blank(),
        text = element_text(family = "Times New Roman"),
        plot.title = element_text(size = 16, face = "bold")) 


print(p_combined)


ggsave("/work/DataScienceExam2024/Data_Science_Exam_S24/plots/feature_importances_COMBINED.jpg", plot = p_combined, width = 10, height = 6, dpi = 300)
