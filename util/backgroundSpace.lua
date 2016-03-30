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

local backgroundSpace, parent = torch.class('backgroundSpace', 'nn.LookupTable')

function backgroundSpace:__init(word2idx, embedding_size)
    local embedding_file = 'util/space.txt' 
    local file_embedding_size = embedding_size
    self.vocab_size = numkeys(word2idx)
    self.word2idx = word2idx
    parent.__init(self, self.vocab_size, embedding_size)		--the lookup table?
    print("loading existing space")
    self.embedding_size = embedding_size
    self.vocab_embedding_file = "util/space.t7"
    w = self:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx)
    --if file_embedding_size ~= embedding_size then
    --    w = torch.mm(w, torch.rand(file_embedding_size, embedding_size))
    --end
    self.weight = w:contiguous()
    print(self.weight[1][1])
    torch.save(self.vocab_embedding_file, self.weight)
    print("loaded space")
end

function backgroundSpace:parseEmbeddingFile(embedding_file, file_embedding_size, word2idx)
    local word_lower2idx = {}
    local loaded = {}
    print(self.vocab_size,file_embedding_size)
    local weight = torch.Tensor(self.vocab_size, file_embedding_size)
    for word, idx in pairs(word2idx) do
        --word_lower2idx[word:lower()] = idx
        word_lower2idx[word] = idx
    end

    for line in io.lines(embedding_file) do
        local parts = split(line, " ")
        local word = parts[1]
        if word_lower2idx[word] then
            local idx = word_lower2idx[word]
            for i=2, #parts do
                weight[idx][i-1] = tonumber(parts[i])
            end
            loaded[word] = true
        end
    end
    for word, idx in pairs(word2idx) do
        if not loaded[word] then
            print("Not loaded: " .. word)
            for i=1, file_embedding_size do
                weight[idx][i] = torch.normal(0, 0.9) --better way to do this?
            end
        end
    end
    return weight
end

