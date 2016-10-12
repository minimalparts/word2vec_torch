local utils = torch.class('utils')

local dump = function(vec)
    vec = vec:view(vec:nElement())
    local t = {}
    for i=1,vec:nElement() do
        t[#t+1] = string.format('%.4f', vec[i])
    end
    return table.concat(t, '\t')
end

function utils:print_semantic_space()
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.word_vecs.weight:double())
    end
    for word,_ in pairs(self.vocab) do
	vec=self.word_vecs_norm[self.word2index[word]]
	vec:resize(vec:size(1),1)				--can only transpose vector with dim 2, so resize
	vec=vec:t()
	io.write(word,"\t",dump(vec),"\n")
    end
end
