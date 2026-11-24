# Keep the stable install path so an already-running shell survives upgrades.
set -g __splash_omni_completion_source (builtin realpath -s -- (status filename))

function __splash_omni_models
    # The helper beside the script's current target, found again each time.
    set -l helper (path dirname -- (path resolve -- $__splash_omni_completion_source))/models
    test -x $helper
    and $helper
end

# As in Bash and Zsh: only the value of --model, and only after `splash-omni serve`.
function __splash_omni_model_value
    set -l words (commandline -opc)
    test "$words[2]" = serve
    and not string match -qr -- '^-[^=]*$' (commandline -ct)
end

complete -c splash-omni -f
complete -c splash-omni -n 'test (count (commandline -opc)) -eq 1' -a 'serve ui'
complete -c splash-omni -n __splash_omni_model_value -l model -x -a '(__splash_omni_models)'
