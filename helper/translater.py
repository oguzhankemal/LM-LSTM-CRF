from googletrans import Translator
import codecs
import copy 

#https://console.developers.google.com/apis/api/translate.googleapis.com/metrics?project=beaming-age-245715&pageState=(%22duration%22:(%22groupValue%22:%22P1D%22))
from google.cloud import translate
# If you don't specify credentials when constructing the client, the
# client library will look for credentials in the environment.
translate_client = translate.Client.from_service_account_json('D:/Program Files/GoogleCloud/beaming-age-245715-13fe12d17275.json')
# The source language
source = 'en'
# The target language
target = 'tr'
# The output format
format = 'text' #text or html(default)
# Translates some text test
#translation = translate_client.translate(
    #'Hello World',
    #format_=format,
    #target_language=target,
    #source_language=source)

#print(translation['translatedText'])

#train_file = 'D:/PythoProjects/Datasets/conll003/conll003-englishversion/train.txt'
#dev_file='D:/PythoProjects/Datasets/conll003/conll003-englishversion/valid.txt'
input_file='D:/PythoProjects/Datasets/conll003/conll003-englishversion/valid.txt'
output_file='D:/PythoProjects/Datasets/conll003/conll003-englishversion/valid_tr.txt'

def read_corpus(lines):
    with open(output_file, 'w', encoding='utf-8') as fout:    
        lineText = ""
        lineTextTemp = ""
        tagText = ""
        tagTextTemp = ""
        previousTag = "O"
        fout.write('-DOCSTART- -X- -X- O\n\n')
        fout.write('')
    
        lineTags = []
        for line in lines:
            if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
                line = line.rstrip('\n').split()
                if (len(lineTextTemp) > 0) and line[0] != "." and line[0] != ",":
                    lineTextTemp += " "
                lineTextTemp += line[0]
                a = line[-1]
                if line[-1].startswith('B-') or line[-1].startswith('O'):
                    if len(tagTextTemp) > 0:
                        tagText = copy.deepcopy(tagTextTemp)
                        translation = translate_client.translate(
                            tagText,
                            format_ = format,
                            target_language=target,
                            source_language=source)
                        newTags = translation['translatedText'].rstrip('\n').split()
                        counter = 1
                        for newTag in newTags:
                            tagBegin = 'B-'
                            if counter > 1:
                                tagBegin = 'I-'
                            lineTags.append(newTag + ' ' + tagBegin + previousTag)
                            print(newTag + ' ' + tagBegin + previousTag)
                            counter = counter + 1
                        
                    if line[-1].startswith('B-'):
                        tagTextTemp = line[0]
                        
                        if 'PER' in line[-1]:
                            previousTag = 'PER'
                        elif 'LOC' in line[-1]:
                            previousTag = 'LOC'
                        elif 'ORG' in line[-1]:
                            previousTag = 'MISC'
                        elif 'MISC' in line[-1]:
                            previousTag = 'MISC'
                    else:
                        tagTextTemp = ''
                        previousTag = 'O'
                    
                
                elif line[-1].startswith('I-'):
                    if (len(tagTextTemp) > 0) and line[0] != "." and line[0] != ",":
                        tagTextTemp += " "
                    tagTextTemp += line[0]
            
            
            elif (line.isspace() and len(lineTextTemp) > 0):
                if len(tagTextTemp) > 0:
                    tagText = copy.deepcopy(tagTextTemp)
                    translation = translate_client.translate(
                        tagText,
                        format_ = format,
                        target_language=target,
                        source_language=source)
                    newTags =translation['translatedText'].rstrip('\n').split()
                    counter = 1
                    for newTag in newTags:
                        tagBegin = 'B-'
                        if counter > 1:
                            tagBegin = 'I-'
                        lineTags.append(newTag + ' ' + tagBegin + previousTag)
                        print(newTag + ' ' + tagBegin + previousTag)
                        counter = counter + 1
                lineText = copy.deepcopy(lineTextTemp)
                translation = translate_client.translate(
                    lineText,
                    format_=format,
                    target_language=target,
                    source_language=source)
                targetLanguageText = translation['translatedText']
                targetLanguageWords = targetLanguageText.rstrip('\n').split()
                for targetLanguageWord in targetLanguageWords:
                    if targetLanguageWord != '-' and targetLanguageWord != '.'  and targetLanguageWord != ',':
                        currentTag = 'O'
                        currentWord = ''
                        for lineTag in lineTags:
                            lineTagParts = lineTag.rstrip('\n').split()
                            if lineTagParts[0] in targetLanguageWord:
                                currentWord = lineTag
                                break
                        #matching = [s for s in lineTags if any(targetLanguageWord in s for xs in lineTags)]
                        if len(currentWord) > 0:
                            currentWord = currentWord.rstrip('\n').split()
                            currentTag = currentWord[-1]
                        fout.write(targetLanguageWord + ' ' + currentTag + '\n')
                        #print(targetLanguageWord + ' ' + currentTag)
            
                fout.write('\n')
                lineTextTemp = ""
                lineTags = []
                tagTextTemp = ''
                print('space ----------------------------------')
        
    

        return lineTags

    # load corpus

print('loading corpus')

with codecs.open(input_file, 'r', 'utf-8') as f:
    input_lines = f.readlines()

lineTags = read_corpus(input_lines)

print(lineTags)