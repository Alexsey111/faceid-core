# evaluation — автономный recognition eval-harness (без DB/Redis/HTTP).
# Метрики точности 1:1 (FAR/FRR/TAR@FAR/EER/AUC/ROC) и 1:N (rank-k/CMC)
# на датасете лиц. Переиспользует ML-компоненты app/ml/*, но не рантайм-сервисы.