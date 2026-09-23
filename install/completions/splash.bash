# Keep the stable install path so an already-running shell survives upgrades.
case ${BASH_SOURCE[0]} in
    /*) _splash_completion_source=${BASH_SOURCE[0]} ;;
    *) _splash_completion_source=$PWD/${BASH_SOURCE[0]} ;;
esac

_splash() {
    local cur prev prefix value_prefix= source directory target model i
    COMPREPLY=()
    cur=${COMP_WORDS[COMP_CWORD]}
    prev=${COMP_WORDS[COMP_CWORD-1]}
    if [[ $COMP_CWORD -eq 1 ]]; then
        COMPREPLY=($(compgen -W 'serve claude codex opencode hermes pi' -- "$cur"))
        return 0
    fi
    [[ ${COMP_WORDS[1]} == serve ]] || return 0
    for ((i=2; i<COMP_CWORD; i++)); do
        [[ ${COMP_WORDS[i]} == -- ]] && return 0
    done
    if [[ $cur == --model=* ]]; then
        prefix=${cur#--model=}
        [[ ${COMP_WORDBREAKS-} == *=* ]] || value_prefix=--model=
    elif [[ $prev == --model ]]; then
        prefix=$cur
        [[ $cur == = ]] && prefix=
    elif [[ $prev == = && $COMP_CWORD -ge 3 &&
            ${COMP_WORDS[COMP_CWORD-2]} == --model ]]; then
        prefix=$cur
    else
        return 0
    fi

    source=$_splash_completion_source
    while :; do
        [[ -f $source ]] || return 0
        directory=$(CDPATH= cd -P "$(dirname "$source")" 2>/dev/null && pwd -P) || return 0
        source=$directory/${source##*/}
        [[ -L $source ]] || break
        target=$(readlink "$source" 2>/dev/null) || return 0
        case $target in
            /*) source=$target ;;
            *) source=$directory/$target ;;
        esac
    done
    [[ -x $directory/models ]] || return 0
    while IFS= read -r model; do
        COMPREPLY[${#COMPREPLY[@]}]=$value_prefix$model
    done < <("$directory/models" "$prefix")
}

complete -F _splash splash
