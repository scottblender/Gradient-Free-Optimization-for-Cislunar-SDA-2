function value = study_hash(input, mode)
% SHA-256 fingerprint of a file or JSON-serializable settings/data.
% JSON mode is for finite numeric data and plain structs, not solver objects.
if nargin < 2, mode = "json"; end
md = java.security.MessageDigest.getInstance('SHA-256');
if mode == "file"
    fid = fopen(input, 'rb');
    assert(fid >= 0, 'Cannot open file: %s', input);
    cleanup = onCleanup(@() fclose(fid));
    while true
        bytes = fread(fid, 1048576, '*uint8');
        if isempty(bytes), break; end
        md.update(typecast(bytes(:), 'int8'));
    end
else
    bytes = unicode2native(jsonencode(input), 'UTF-8');
    md.update(typecast(uint8(bytes(:)), 'int8'));
end
bytes = typecast(int8(md.digest()), 'uint8');
value = string(lower(reshape(dec2hex(bytes,2).',1,[])));
end
