from random import choice, randint

def get_response(user_input: str) -> str:
    lowered:str = user_input.lower()
    
    
    if lowered == '':
        return 'Speak up bruh'
    elif 'hello' in lowered:
        return 'wsg gang'
    elif 'roll dice' in lowered:
        return f'thas a {randint(1, 6)}'
    else:
        return 'im a fraud :('