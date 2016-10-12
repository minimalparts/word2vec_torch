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

local backgroundSpace, parent = torch.class('backgroundSpace', 'nn.LookupTable')

function backgroundSpace:__init(word2idx, embedding_size, background)
    local embedding_file = background 
    local file_embedding_size = embedding_size
    self.vocab_size = numkeys(word2idx)
    self.word2idx = word2idx
    parent.__init(self, self.vocab_size, embedding_size)		--the lookup table?
    --print("loading existing space")
    self.embedding_size = embedding_size
    self.vocab_embedding_file = background..".t7"
    if file_exists(self.vocab_embedding_file)
    then
	self.weight = torch.load(self.vocab_embedding_file)
    else
        w = self:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx)
        --if file_embedding_size ~= embedding_size then
        --    w = torch.mm(w, torch.rand(file_embedding_size, embedding_size))
        --end
        self.weight = w:contiguous()
        torch.save(self.vocab_embedding_file, self.weight)
    end
    --print("loaded space")
end

function backgroundSpace:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx, gpu)
    local word_lower2idx = {}
    local loaded = {}
    local weight = torch.Tensor()
    --print(self.vocab_size,file_embedding_size)
    if gpu == 1 then
        weight = torch.CudaTensor(self.vocab_size, file_embedding_size)
    else
        weight = torch.Tensor(self.vocab_size, file_embedding_size)
    end
    -- print(weight)
    --print(numkeys(word2idx))
    for word, idx in pairs(word2idx) do
	--print(word,idx)
        --word_lower2idx[word:lower()] = idx
        word_lower2idx[word] = idx
    end

    for line in io.lines(embedding_file) do
        local parts = split(line, " ")
        local word = parts[1]
        if word_lower2idx[word] then
            local idx = word_lower2idx[word]
            --print(word,idx)
            for i=2, #parts do
                weight[idx][i-1] = tonumber(parts[i])
            end
            loaded[word] = true
        end
    end
    for word, idx in pairs(word2idx) do
        if not loaded[word] then
            --print("Not loaded: " .. word)
            for i=1, file_embedding_size do
                weight[idx][i] = torch.normal(0, 0.9) --better way to do this?
            end
        end
    end
    return weight
end

