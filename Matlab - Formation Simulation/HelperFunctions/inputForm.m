function s = inputForm(A, name)
%s = inputForm(A, name)
% Create assignment code from disp output
%
% Inputs
%   A: anything disp can handle
%   name: variable A is assigned to
% Output
%   s:  string with expression 'name = A;'
if nargin == 0 % demo
   A = randn(4);
   name = 'Amat';
end
nl = sprintf('\n');
s = evalc('disp(A)');
if strfind(s, 'Columns')
    disp('cannot yet handle that many columns')
    disp('you can try to enlarge the Command Window')
    error('');
end
s = strrep(s, '[', ''); 
s = strrep(s, ']', '');
s = strrep(s, nl, ['; ...', nl]);
s = [name, ' = [ ...', nl, s, '];'];