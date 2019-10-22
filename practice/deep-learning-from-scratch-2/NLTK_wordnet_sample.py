# =============================================================================
# conda install nltk
# =============================================================================

from nltk.corpus import wordnet


# WordNet에서 동의어 얻기
wordnet.synset('car')

# =============================================================================
# LookupError: 
# **********************************************************************
#   Resource wordnet not found.
#   Please use the NLTK Downloader to obtain the resource:
# 
# import nltk
# nltk.download('wordnet')
#   
#   For more information see: https://www.nltk.org/data.html
# 
#   Attempted to load corpora/wordnet
# 
#   Searched in:
#     - 'C:\\Users\\heaven/nltk_data'
#     - 'C:\\Users\\heaven\\Anaconda3\\envs\\tensor\\nltk_data'
#     - 'C:\\Users\\heaven\\Anaconda3\\envs\\tensor\\share\\nltk_data'
#     - 'C:\\Users\\heaven\\Anaconda3\\envs\\tensor\\lib\\nltk_data'
#     - 'C:\\Users\\heaven\\AppData\\Roaming\\nltk_data'
#     - 'C:\\nltk_data'
#     - 'D:\\nltk_data'
#     - 'E:\\nltk_data'
# **********************************************************************
# =============================================================================




# =============================================================================
# python3 버전에 맞게 갱신한 버전 공부하려면
# https://www.nltk.org/book/ 들어가서 확인!
# =============================================================================
