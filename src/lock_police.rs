use rayon::vec;
use std::sync::MutexGuard;

use super::*;
use crate::interface::*;
use crate::utility::*;
use crate::card::*;
use crate::card_to_string;

type ErrorVec = Vec<(Vec<Action>, [u16; 2])>;

pub fn police_locks_strings(tree: MutexGuard<ActionTree>, card_config: &CardConfig) -> Vec<String>
{
    let new_tree = tree.clone();
    let new_config = card_config.clone();

    let toy_game = PostFlopGame::with_config(new_config, new_tree).unwrap();
    
    let mut error_vec = police_locks(toy_game);

    // collapsing the vecs

    for chance in 0..2
    {
        let mut new_vec: ErrorVec = vec![];

        // vecs with same indexing
        let mut generics = vec![] as Vec<(Vec<Action>, [u16; 2], bool, u8)>;

        for error in &error_vec
        {
            let error_data = action_vec_chance_data(&error.0);
            let mut generic_vec: Vec<Action>;

            if error_data.0 != 0 && chance == 0
            {
                generic_vec = error.0.clone();
                generic_vec[error_data.1 as usize] = Action::Generic;
            }
            else if error_data.0 == 2 && chance == 1
            {
                generic_vec = error.0.clone();
                generic_vec[error_data.2 as usize] = Action::Generic;
            }
            else 
            {
                continue
            }

            let finder = generics.iter().position(|x| x.0 == generic_vec);

            if finder.is_some()
            {
                let id = finder.unwrap();

                if generics[id].1 == error.1
                {
                    generics[id].3 += 1;
                    continue;
                }
                else 
                {
                    generics[id].2 = false;
                }
            }
            else 
            {
                generics.push((generic_vec, error.1, true, 1));
            }
        }

        for error in &error_vec
        {
            let error_data = action_vec_chance_data(&error.0);
            let mut generic_vec: Vec<Action>;

            if error_data.0 != 0 && chance == 0
            {
                generic_vec = error.0.clone();
                generic_vec[error_data.1 as usize] = Action::Generic;
            }
            else if error_data.0 == 2 && chance == 1
            {
                generic_vec = error.0.clone();
                generic_vec[error_data.2 as usize] = Action::Generic;
            }
            else 
            {
                new_vec.push(error.clone());
                continue
            }

            let finder = generics.iter().find(|x| x.0 == generic_vec);

            if 
                finder.is_some() && 
                finder.unwrap().2 &&
                (
                    (finder.unwrap().3 == 49 && chance == 0) ||
                    (finder.unwrap().3 == 48 && chance == 1)
                )
            {
                let generic_error = (generic_vec, error.1);

                if new_vec.contains(&generic_error)
                {
                    continue
                }
                else
                {
                    new_vec.push(generic_error);
                }
            }
            else
            {
                new_vec.push(error.clone());
            }
        }

        error_vec = new_vec;
    }



    // actually making strings

    let mut strings: Vec<String> = vec![];

    for error in &error_vec
    {
        let mut new_string = "".to_owned();

        if error.0.len() == 0
        {
            new_string += "ROOT :";
        }

        for i in 0..error.0.len()
        {
            let mut text = match error.0[i] {
                Action::Chance(x) => {
                    let ret = card_to_string(x);

                    if ret.is_ok() {
                        ret.unwrap()
                    }
                    else {
                        "Unknown card id".to_owned()
                    }
                },
                Action::Generic => "ANY CARD".to_owned(),

                _ => error.0[i].to_action_string()
            };

            if i < error.0.len() - 1
            {
                text += " > ";
            }
            else
            {
                text += " :"
            }

            new_string += &text;
        }

        if error.1[0] > 0
        {
            new_string += &format!(" {} underflows", error.1[0]);
        }
        if error.1[1] > 0
        {
            new_string += &format!(" {} overflows", error.1[1]);
        }

        strings.push(new_string);
    }

    strings
}

pub fn police_locks(game: PostFlopGame) -> ErrorVec
{
    let history_base: Vec<Action> = vec![];
    let error_vec: Vec<(Vec<Action>, [u16; 2])> = vec![];

    let root = game.root();

    game.verify_locks_recursive(root, &history_base, error_vec)
}

impl PostFlopGame
{
    pub fn verify_locks(
        &self
    ) -> ErrorVec
    {
        let history_base: Vec<Action> = vec![];
        let error_vec: Vec<(Vec<Action>, [u16; 2])> = vec![];

        let root = self.root();

        self.verify_locks_recursive(root, &history_base, error_vec)
    }

    fn verify_locks_recursive(
        &self, 
        node_mgl: MutexGuardLike<PostFlopNode>, 
        history: &Vec<Action>, 
        mut error_vec: ErrorVec
    ) -> ErrorVec
    {
        if node_mgl.is_chance() {

        }
        else if node_mgl.is_terminal()
        {
            return error_vec
        }
        else
        {
            let my_end_range = node_mgl.my_end_range(self);
            let my_end_limit = node_mgl.my_end_limit(self);

            let mut overdrives: u16 = 0;
            let mut underdrives: u16 = 0;
            
            for i in 0..RANGESIZE
            {
                let mut min_lock: f32 = 0.0;
                let mut max_lock: f32 = 0.0;

                for j in 0..node_mgl.num_actions()
                {
                    let curr_id = i + j * RANGESIZE;

                    if my_end_limit[curr_id] == -1
                    {
                        max_lock += my_end_range[curr_id];
                    }
                    else if my_end_limit[curr_id] == 1
                    {
                        max_lock += 1.0;
                        min_lock += my_end_range[curr_id];
                    }
                    else 
                    {
                        max_lock += my_end_range[curr_id];
                        min_lock += my_end_range[curr_id];
                    }
                }

                if min_lock > 1.0
                {
                    overdrives += 1;
                }
                if max_lock < 1.0
                {
                    underdrives += 1;
                }
            }


            if overdrives > 0 || underdrives > 0
            {
                error_vec.push((history.clone(), [underdrives, overdrives]));
            }
        }

        for child_mxl in node_mgl.children()
        {
            let child = child_mxl.lock();

            let mut new_history = history.clone();
            new_history.push(child.prev_action);

            error_vec = self.verify_locks_recursive(child, &new_history, error_vec);
        }

        error_vec
    }
}

impl Action
{
    fn to_action_string(&self) -> String
    {
        let result = match self 
        {
            Action::None => "None",
            Action::Fold => "FOLD",
            Action::Check => "CHECK",
            Action::Call => "CALL",
            Action::Bet(x) => &format!("BET {}" , x),
            Action::Raise(x) => &format!("RAISE {}" , x),
            Action::AllIn(x) => &format!("ALL-IN {}" , x),

            Action::Chance(x) => &format!("CHANCE {}" , x),

            Action::Generic => "GENERIC"
        };

        result.to_owned()
    }
}

fn match_generic_action_vec(generic: &Vec<Action>, vector: &Vec<Action>) -> bool
{
    if generic.len() != vector.len()
    {
        return false
    }
    else
    {
        let mut equals = true;

        for i in 0..generic.len()
        {
            if generic[i] != vector[i] && generic[i] != Action::Generic
            {
                equals = false;
                break;
            }
        }

        equals
    }
}

fn action_vec_chance_data(vector: &Vec<Action>) -> (u8, u8, u8)
{
    let mut data = (0 as u8, u8::MAX, u8::MAX);

    for i in 0..vector.len()
    {
        match vector[i] {
            Action::Chance(_) | Action::Generic => {
                data.0 += 1;

                if data.0 == 1
                {
                    data.1 = i as u8;
                }
                if data.0 == 2
                {
                    data.2 = i as u8;
                }
            }

            _ => continue
        }
    }

    data
}