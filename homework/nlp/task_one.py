"""
import pkuseg

seg = pkuseg.pkuseg(model_name='medicine')
# Automatically download the domain-specific model.
text = seg.cut('我爱北京天安门')
print(text)

--------------

代码示例1
import thulac

thu1 = thulac.thulac()  #默认模式
text = thu1.cut("我爱北京天安门", text=True)  #进行一句话分词
print(text)
代码示例2
thu1 = thulac.thulac(seg_only=True)  #只进行分词，不进行词性标注
thu1.cut_f("input.txt", "output.txt")  #对input.txt文件内容进行分词，输出到output.txt


-------------------
import jieba
import jieba.posseg #需要另外加载一个词性标注模块

string = '其实大家买手机就是看个心情，没必要比来比去的。'
seg = jieba.posseg.cut(string)


"""

import re

import jieba.posseg
import pkuseg
import thulac

re_sub = re.compile("< ?sub ?> .+?</sub ?>")


def read_data():
    re_line = re.compile("(^\d+) (.*)")
    with open("raw_66.txt", "r", encoding="utf-8") as f:
        lines = [re_line.findall(line)[0] for line in f]
        return lines


def cut_lines():
    """切分句子"""
    seg = pkuseg.pkuseg(model_name='medicine')
    cut_result = []
    for line_num, line_text in read_data():
        ss = seg.cut(line_text)
        cut_line = " ".join([line_num] + ss)
        for s in re_sub.findall(cut_line):  # 跳过<sub></sub>
            cut_line = cut_line.replace(s, s.replace(" ", ''))
        for seq_num, dot in re.findall("(\d+)(\.)", cut_line):
            cut_line = cut_line.replace(seq_num + dot, "{} {}".format(seq_num, dot))
        for blank in re.findall("\s+?\$\$\s?_+\s?", cut_line):
            cut_line = cut_line.replace(blank, blank.replace(" ", ''))
        cut_result.append(cut_line)

    stage1_res = "\n".join(cut_result)
    with open("cut_result.txt", 'w', encoding="utf-8") as f:
        f.write(stage1_res)
    return stage1_res


def mark_word():
    """标记句子"""
    with open("cut_result.txt", 'r', encoding="utf-8") as f:
        cutted_lines = f.readlines()
        lines = [cutted_line.strip().split(" ")[1:] for cutted_line in cutted_lines]
    thu = thulac.thulac(seg_only=False)  # 只进行分词，不进行词性标注
    mark_result = []
    for line_num, line_words in enumerate(lines):
        # print(line_words)
        line_result = []
        for word in line_words:
            cut_word = word
            sub_note = re_sub.findall(word)
            if sub_note:
                for s in sub_note:
                    cut_word = cut_word.replace(s, '')
            thu_ss = thu.cut(cut_word)
            jieba_ss = jieba.posseg.cut(cut_word)
            jieba_ss = [(s.word, s.flag) for s in jieba_ss]
            flag = ''
            if len(thu_ss) == len(jieba_ss) == 1:
                if thu_ss[0][1] == jieba_ss[0][1]:
                    flag = thu_ss[0][1]
                elif thu_ss[0][1] in ("m", "q", "u", "w"):
                    flag = thu_ss[0][1]
                elif jieba_ss[0][1] == "eng":
                    flag = "nx"  # 英文单词
                # else:
                #     flag = thu_ss[0][1]
            if flag == "":
                if jieba_ss[0][1] == "eng":
                    flag = "nx"  # 英文单词
                else:
                    flag = "?[{}|{}]".format(thu_ss[0][1], jieba_ss[0][1])
            # print("word: {}".format(word))
            # print("flag: {}".format(flag))
            # print("thulac cut: {}".format(thu_ss))
            # print("jieba cut: {}".format(jieba_ss))
            # print("\n")
            line_result.append("{}/{}".format(word, flag))
        mark_result.append(" ".join([str(line_num + 1)] + line_result))
    with open("mark_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(mark_result))


if __name__ == "__main__":
    res = cut_lines()
    # print(res)
    mark_word()
