-- Code borrowed and modified from https://github.com/larspars/word-rnn/blob/master/util/GloVeEmbedding.lua
-- IMPORTANT: background space must be space (not tab)-delimited

function split(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

function numkeys(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

local backgroundVocab = torch.class("backgroundVocab")

function backgroundVocab:__init(background)
    self.vocabulary = {}
    background = background:gsub("spaces/","vocabs/")
    if background ~= "none" then 
	    self.vocab_file = background..".vocab.t7"
	    if file_exists(self.vocab_file)
	    then
		self.vocabulary = torch.load(self.vocab_file)
	    else
		-- Add vocab from the background space
		unigram_file = background..".vocab"
		for line in io.lines(unigram_file) do
		    local parts = split(line, " ")
		    local freq = tonumber(parts[2])
		    local word = parts[1]
		    if self.vocabulary[word:lower()] == nil then
		        self.vocabulary[word:lower()] = freq	 
		    else
		        self.vocabulary[word:lower()] = self.vocabulary[word:lower()] + freq
		    end
		end
		torch.save(self.vocab_file, self.vocabulary)
	   end
    end

    c=0
    for word, count in pairs(self.vocabulary) do
        c = c+1
   end
    --print(string.format("Vocab size: %d", c))
    --print("Vocab loaded!")
end


