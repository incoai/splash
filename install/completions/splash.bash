# Keep the stable install path so an already-running shell survives upgrades.
case ${BASH_SOURCE[0]} in
    /*) _splash_completion_source=${BASH_SOURCE[0]} ;;
    *) _splash_completion_source=$PWD/${BASH_SOURCE[0]} ;;
esac

_splash() {
    local cur prev prefix value_prefix= trim= repository variant source directory
    local target model i
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
        # Bash 4 and later split owner/repo:VARIANT at a ':' in COMP_WORDBREAKS
        # into the repository, ':' and the variant typed so far.
        if [[ $cur == : ]]; then
            i=$((COMP_CWORD - 1)) variant=
        elif [[ $prev == : ]]; then
            i=$((COMP_CWORD - 2)) variant=$cur
        else
            return 0
        fi
        repository=${COMP_WORDS[i]}
        if [[ $repository == --model=* ]]; then
            repository=${repository#--model=}
        elif ! [[ ${COMP_WORDS[i-1]} == --model ||
                ${COMP_WORDS[i-1]} == = && ${COMP_WORDS[i-2]} == --model ]]; then
            return 0
        fi
        prefix=$repository:$variant
    fi
    # Readline replaces only the text after the last ':' word break, also where
    # Bash 3.2 keeps owner/repo:VARIANT in one word, so candidates drop what
    # precedes it, as bash-completion's __ltrim_colon_completions does.
    if [[ ${COMP_WORDBREAKS-} == *:* && $prefix == *:* ]]; then
        trim=${prefix%"${prefix##*:}"}
        value_prefix=
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
        COMPREPLY[${#COMPREPLY[@]}]=$value_prefix${model#"$trim"}
    done < <("$directory/models" "$prefix")
}

complete -F _splash splash
