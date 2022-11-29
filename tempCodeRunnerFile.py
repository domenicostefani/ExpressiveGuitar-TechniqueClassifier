metric_names = ['accuracy','f1_weightedmacroavg','confusion_matrix','classification_report','printable_classification_report']
evaluation_metrics, quantized_model_evaluation_metrics = dict.fromkeys(metric_names,[]), dict.fromkeys(metric_names,[])



evaluation_metrics1 = { "accuracy" : [], "f1_weightedmacroavg" : [], "confusion_matrix" : [],"classification_report" : [],"printable_classification_report" : [] }
quantized_model_evaluation_metrics1 = { "accuracy" : [], "f1_weightedmacroavg" : [], "confusion_matrix" : [],"classification_report" : [],"printable_classification_report" : [] }


assert evaluation_metrics == evaluation_metrics1
assert quantized_model_evaluation_metrics == quantized_model_evaluation_metrics1

print('They are in fact equal')