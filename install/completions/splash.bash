# Keep the stable install path so an already-running shell survives upgrades.
case ${BASH_SOURCE[0]} in
    /*) _splash_completion_source=${BASH_SOURCE[0]} ;;
    *) _splash_completion_source=$PWD/${BASH_SOURCE[0]} ;;
esac

_splash() {
    local cur prev prefix value_prefix= trim= repository variant source directory
    local target model i option
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
    # The option whose value this completes and the value typed so far. Bash 4
    # and later split --option=value at an '=' in COMP_WORDBREAKS into the
    # option, '=' and the value, so the word is the '=' until a value is typed.
    if [[ $cur == --*=* ]]; then
        option=${cur%%=*} prefix=${cur#*=}
        [[ ${COMP_WORDBREAKS-} == *=* ]] || value_prefix=$option=
    elif [[ $cur == = ]]; then
        option=$prev prefix=
    elif [[ $prev == = ]]; then
        option=${COMP_WORDS[COMP_CWORD-2]} prefix=$cur
    else
        option=$prev prefix=$cur
    fi
    case $option in
        --model-dir|--draft-model|--mmproj)
            # compgen -f run in this shell, unlike the subshell below, has
            # readline treat these candidates as file names: quoted, with a
            # directory's ending in '/'. Bash 3.2 has no compopt, and complete's
            # filenames option would treat model IDs so too, escaping their ':'.
            compgen -f -- "$prefix" > /dev/null
            while IFS= read -r target; do
                COMPREPLY[${#COMPREPLY[@]}]=$value_prefix$target
            done < <(compgen -f -- "$prefix")
            return 0
            ;;
    esac
    if [[ $option != --model ]]; then
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
